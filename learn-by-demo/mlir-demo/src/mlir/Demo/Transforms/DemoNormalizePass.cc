#include "mlir/Demo/IR/DemoOps.h"
#include "mlir/Demo/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace demo {

#define GEN_PASS_DEF_DEMONORMALIZEPASS
#include "mlir/Demo/Transforms/Passes.h.inc"

}  // namespace demo
}  // namespace mlir

namespace {

struct ConvertPrintToPrintImpl
    : public mlir::OpRewritePattern<mlir::demo::PrintOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(
      mlir::demo::PrintOp op, mlir::PatternRewriter& rewriter) const final {
    ::llvm::SmallVector<::mlir::Value, 4> operands{op->getOperands()};
    rewriter.create<mlir::demo::PrintImplOp>(
        op->getLoc(), ::llvm::SmallVector<::mlir::Type, 4>{}, operands);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct DemoNormalizePass
    : mlir::demo::impl::DemoNormalizePassBase<DemoNormalizePass> {
  using DemoNormalizePassBase::DemoNormalizePassBase;
  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::demo::DemoDialect>();
    target.addIllegalOp<mlir::demo::PrintOp>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ConvertPrintToPrintImpl>(&getContext());
    if (mlir::failed(applyPartialConversion(getOperation(), target,
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
