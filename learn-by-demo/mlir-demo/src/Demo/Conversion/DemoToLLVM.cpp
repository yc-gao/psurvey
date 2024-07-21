#include "DemoToLLVM.h"

#include <memory>

#include "Demo/IR/DemoOps.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

namespace {

struct PrintOpLowering : public mlir::OpRewritePattern<mlir::demo::PrintOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(
      mlir::demo::PrintOp op, mlir::PatternRewriter &rewriter) const final {
    auto *context = rewriter.getContext();
    auto memRefType = llvm::cast<mlir::MemRefType>((*op->operand_type_begin()));
    auto memRefShape = memRefType.getShape();
    auto loc = op->getLoc();

    mlir::ModuleOp parentModule = op->getParentOfType<mlir::ModuleOp>();

    // Get a symbol reference to the printf function, inserting it if necessary.
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    mlir::Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec", llvm::StringRef("%f \0", 4), parentModule);
    mlir::Value newLineCst = getOrCreateGlobalString(
        loc, rewriter, "nl", llvm::StringRef("\n\0", 2), parentModule);

    // Create a loop for each of the dimensions within the shape.
    llvm::SmallVector<mlir::Value, 4> loopIvs;
    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
      auto lowerBound = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      auto upperBound =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, memRefShape[i]);
      auto step = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step);
      for (mlir::Operation &nested : *loop.getBody()) rewriter.eraseOp(&nested);
      loopIvs.push_back(loop.getInductionVar());

      // Terminate the loop body.
      rewriter.setInsertionPointToEnd(loop.getBody());

      // Insert a newline after each of the inner dimensions of the shape.
      if (i != e - 1) {
        rewriter.create<mlir::LLVM::CallOp>(loc, getPrintfType(context),
                                            printfRef, newLineCst);
      }
      rewriter.create<mlir::scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Generate a call to printf for the current element of the loop.
    auto printOp = llvm::cast<mlir::demo::PrintOp>(op);
    auto elementLoad =
        rewriter.create<mlir::memref::LoadOp>(loc, printOp.getInput(), loopIvs);
    rewriter.create<mlir::LLVM::CallOp>(
        loc, getPrintfType(context), printfRef,
        llvm::ArrayRef<mlir::Value>({formatSpecifierCst, elementLoad}));

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return mlir::success();
  }

 private:
  static mlir::LLVM::LLVMFunctionType getPrintfType(
      mlir::MLIRContext *context) {
    auto llvmI32Ty = mlir::IntegerType::get(context, 32);
    auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(context);
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy,
                                                        /*isVarArg=*/true);
    return llvmFnType;
  }

  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static mlir::FlatSymbolRefAttr getOrInsertPrintf(
      mlir::PatternRewriter &rewriter, mlir::ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf"))
      return mlir::SymbolRefAttr::get(context, "printf");

    // Insert the printf function into the body of the parent module.
    mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "printf",
                                            getPrintfType(context));
    return mlir::SymbolRefAttr::get(context, "printf");
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static mlir::Value getOrCreateGlobalString(mlir::Location loc,
                                             mlir::OpBuilder &builder,
                                             mlir::StringRef name,
                                             mlir::StringRef value,
                                             mlir::ModuleOp module) {
    // Create the global at the entry of the module.
    mlir::LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<mlir::LLVM::GlobalOp>(
          loc, type, /*isConstant=*/true, mlir::LLVM::Linkage::Internal, name,
          builder.getStringAttr(value),
          /*alignment=*/0);
    }

    // Get the pointer to the first character in the global string.
    mlir::Value globalPtr =
        builder.create<mlir::LLVM::AddressOfOp>(loc, global);
    mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI64Type(), builder.getIndexAttr(0));
    return builder.create<mlir::LLVM::GEPOp>(
        loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
        global.getType(), globalPtr, llvm::ArrayRef<mlir::Value>({cst0, cst0}));
  }
};

struct DemoToLLVMLoweringPass0
    : public mlir::PassWrapper<DemoToLLVMLoweringPass0,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(
      mlir::DialectRegistry &registry) const override final {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect, mlir::arith::ArithDialect>();
  }
  void runOnOperation() override final {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect,
                           mlir::scf::SCFDialect, mlir::arith::ArithDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<PrintOpLowering>(&getContext());

    if (failed(mlir::applyPartialConversion(getOperation(), target,
                                            std::move(patterns))))
      signalPassFailure();
  }
};

struct DemoToLLVMLoweringPass1
    : public mlir::PassWrapper<DemoToLLVMLoweringPass1,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(
      mlir::DialectRegistry &registry) const override final {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }
  void runOnOperation() override final {
    mlir::LLVMConversionTarget target(getContext());
    target.addLegalOp<mlir::ModuleOp>();

    mlir::LLVMTypeConverter typeConverter(&getContext());

    mlir::RewritePatternSet patterns(&getContext());
    mlir::populateSCFToControlFlowConversionPatterns(patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter,
                                                         patterns);
    mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(mlir::applyFullConversion(getOperation(), target,
                                         std::move(patterns))))
      signalPassFailure();
  }
};

};  // namespace

namespace mlir {
namespace demo {

void AddPassesDemoToLLVM(mlir::PassManager &pm) {
  pm.addPass(std::make_unique<DemoToLLVMLoweringPass0>());
  pm.addPass(std::make_unique<DemoToLLVMLoweringPass1>());
}

}  // namespace demo
}  // namespace mlir
