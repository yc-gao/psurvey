; ModuleID = 'test'
source_filename = "test"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

define i32 @fib(i32 %AnArg) {
EntryBlock:
  %cond = icmp sle i32 %AnArg, 2
  br i1 %cond, label %return, label %recurse

return:                                           ; preds = %EntryBlock
  ret i32 1

recurse:                                          ; preds = %EntryBlock
  %arg = sub i32 %AnArg, 1
  %fibx1 = tail call i32 @fib(i32 %arg)
  %arg1 = sub i32 %AnArg, 2
  %fibx2 = tail call i32 @fib(i32 %arg1)
  %addresult = add i32 %fibx1, %fibx2
  ret i32 %addresult
}
