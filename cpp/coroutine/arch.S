.globl CoSwitch
.type	CoSwitch, @function
CoSwitch:
    push %rdi
    push %rsi
    push %rdx
    push %rcx
    push %r8
    push %r9

    push %rax
    push %rbx
    push %rbp
    push %r10
    push %r12
    push %r13
    push %r14
    push %r15
    
    mov %rsp, (%rdi)
    mov (%rsi), %rsp
    
    pop %r15
    pop %r14
    pop %r13
    pop %r12
    pop %r10
    pop %rbp
    pop %rbx
    pop %rax

    pop %r9
    pop %r8
    pop %rcx
    pop %rdx
    pop %rsi
    pop %rdi
    ret

.globl CoInit
.type	CoInit, @function
CoInit:
    mov (%rdi), %rax
    mov %rsi, -8(%rax)
    mov %rdx, -16(%rax)
    sub $120, %rax
    mov %rax, (%rdi)

