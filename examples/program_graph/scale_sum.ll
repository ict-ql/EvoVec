define i32 @scale_sum(i32 %a, i32 %b) {
entry:
  %x = add i32 %a, %b
  %y = mul i32 %x, 4
  ret i32 %y
}
