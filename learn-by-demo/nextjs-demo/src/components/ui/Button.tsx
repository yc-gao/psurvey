import React from 'react';
import { cva } from 'class-variance-authority'
import type { VariantProps } from "class-variance-authority";

const buttonVariants = cva('border rounded font-bold', {
    variants: {
        variant: {
            primary: 'bg-blue-500 hover:bg-blue-700 text-white',
            secondary: 'bg-gray-500 hover:bg-gray-700 text-white',
        },
        size: {
            sm: "px-2 py-1 text-sm",
            md: "px-4 py-2 text-base",
            lg: "px-6 py-3 text-lg",
        },
    },
    defaultVariants: {
        variant: 'primary',
        size: 'md',
    },
})
export type ButtonProps = React.ComponentProps<'button'> & VariantProps<typeof buttonVariants>

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(({ children, variant, size, ...props }, ref) => {
    return <button ref={ref} className={buttonVariants({ variant, size })} {...props}>{children}</button>
})
Button.displayName = 'Button'