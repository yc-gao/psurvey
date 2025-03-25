import Link from 'next/link';

export default function() {
    return (
        <div className="w-screen h-dvh flex justify-center pt-12 md:pt-0 items-start md:items-center">
            <div className="flex flex-col justify-start items-stretch gap-12">
                <div className='w-full flex flex-col gap-2 px-12'>
                    <h3 className="text-xl text-center font-semibold">Sign in</h3>
                    <p className="text-sm text-gray-500">
                        Use your email and password to sign in.
                    </p>
                </div>
                <form action="/" className='w-full flex flex-col justify-start items-stretch gap-4'>
                    <div className='flex flex-col items-stretch gap-2'>
                        <p className='text-zinc-600'>Email Address</p>
                        <input
                            type="text"
                            placeholder="user@acme.com"
                            className='h-10 w-full rounded-md border bg-background px-3 py-2'
                        />
                    </div>
                    <div className='flex flex-col items-stretch gap-2'>
                        <p className='text-zinc-600'>Password</p>
                        <input
                            type="password"
                            className='h-10 w-full rounded-md border bg-background px-3 py-2'
                        />
                    </div>
                    <input
                        type="submit"
                        value="Sign in"
                        className='h-10 w-full rounded-md bg-gray-950 text-gray-50 hover:cursor-pointer'
                    />
                    <p className='text-center text-sm text-gray-600'>
                        {"Don't have an account? "}
                        <Link
                            href="/signup"
                            className='font-semibold text-gray-800 hover:underline'
                        >
                            Sign up
                        </Link>
                        {" for free."}
                    </p>
                </form>
            </div>
        </div>
    );
}
