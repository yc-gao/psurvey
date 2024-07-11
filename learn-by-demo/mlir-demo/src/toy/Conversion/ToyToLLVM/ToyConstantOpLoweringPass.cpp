#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "toy/IR/ToyOps.h"

namespace {

static mlir::MemRefType convertTensorToMemRef(mlir::RankedTensorType type) {
  return mlir::MemRefType::get(type.getShape(), type.getElementType());
}

static mlir::Value insertAllocAndDealloc(mlir::MemRefType type,
                                         mlir::Location loc,
                                         mlir::PatternRewriter &rewriter) {
  auto alloc = rewriter.create<mlir::memref::AllocOp>(loc, type);

  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  auto dealloc = rewriter.create<mlir::memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

struct ToyConstantOpLowering
    : public mlir::OpRewritePattern<mlir::toy::ToyConstantOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(
      mlir::toy::ToyConstantOp op,
      mlir::PatternRewriter &rewriter) const final {
    mlir::DenseElementsAttr constantValue = op.getValue();
    mlir::Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = llvm::cast<mlir::RankedTensorType>(op.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    llvm::SmallVector<mlir::Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
               0, *std::max_element(valueShape.begin(), valueShape.end())))
        constantIndices.push_back(
            rewriter.create<mlir::arith::ConstantIndexOp>(loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0));
    }

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    llvm::SmallVector<mlir::Value, 2> indices;
    auto valueIt = constantValue.value_begin<mlir::FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<mlir::affine::AffineStoreOp>(
            loc, rewriter.create<mlir::arith::ConstantOp>(loc, *valueIt++),
            alloc, llvm::ArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return mlir::success();
  }
};

struct ToyPrintOpLowering
    : public mlir::OpConversionPattern<mlir::toy::ToyPrintOp> {
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      mlir::toy::ToyPrintOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const final {
    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });
    return mlir::success();
  }
};

struct ToyConstantOpLoweringPass
    : public mlir::PassWrapper<ToyConstantOpLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyConstantOpLoweringPass);
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect, mlir::arith::ArithDialect,
                    mlir::affine::AffineDialect>();
  }
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::BuiltinDialect, mlir::memref::MemRefDialect,
                           mlir::arith::ArithDialect,
                           mlir::affine::AffineDialect>();
    target.addDynamicallyLegalOp<mlir::toy::ToyPrintOp>(
        [](mlir::toy::ToyPrintOp op) {
          return llvm::none_of(op->getOperandTypes(), [](mlir::Type type) {
            return llvm::isa<mlir::TensorType>(type);
          });
        });

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ToyConstantOpLowering, ToyPrintOpLowering>(
        patterns.getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // namespace

namespace mlir {
namespace toy {

std::unique_ptr<mlir::Pass> createToyConstantOpLoweringPass() {
  return std::make_unique<ToyConstantOpLoweringPass>();
}

}  // namespace toy
}  // namespace mlir
