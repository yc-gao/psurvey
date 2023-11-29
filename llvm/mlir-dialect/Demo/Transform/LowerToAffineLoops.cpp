#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Demo.h"
#include "DemoConversions.h"

namespace mlir {
namespace demo {

class ConstantOpLowering
    : public mlir::OpRewritePattern<mlir::demo::ConstantOp> {
  mlir::MemRefType convertTensorToMemRef(mlir::TensorType type) const {
    return MemRefType::get(type.getShape(), type.getElementType());
  }

  mlir::Value insertAllocAndDealloc(MemRefType type, Location loc,
                                    PatternRewriter &rewriter) const {
    auto alloc = rewriter.create<mlir::memref::AllocOp>(loc, type);
    auto *parentBlock = alloc->getBlock();
    alloc->moveBefore(&parentBlock->front());

    auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
    dealloc->moveBefore(&parentBlock->back());
    return alloc;
  }

public:
  using OpRewritePattern<mlir::demo::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::demo::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    DenseElementsAttr constantValue = op.getValue();
    mlir::TensorType tensorType = op.getType().cast<mlir::TensorType>();
    mlir::MemRefType memRefType = convertTensorToMemRef(tensorType);
    mlir::Value alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    ::llvm::ArrayRef<int64_t> valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
               0, *std::max_element(valueShape.begin(), valueShape.end())))
        constantIndices.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.value_begin<FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<AffineStoreOp>(
            loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
            llvm::ArrayRef(indices));
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

    storeElements(/*dimension=*/0);
    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

class DemoToAffineLoweringPass
    : public mlir::PassWrapper<DemoToAffineLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::AffineDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect>();
  }
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect, arith::ArithDialect,
                           func::FuncDialect, memref::MemRefDialect>();
    target.addIllegalDialect<mlir::demo::DemoDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantOpLowering>(&getContext());

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createLowerToAffinePass() {
  return std::make_unique<DemoToAffineLoweringPass>();
}

} // namespace demo
} // namespace mlir
