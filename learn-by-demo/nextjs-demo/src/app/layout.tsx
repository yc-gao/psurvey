import React from "react";
import type { Metadata } from 'next'

import "./globals.css";

export const metadata: Metadata = {
  title: "Next.js Demo",
  description: "A demo of Next.js with Tailwind CSS and Heroicons",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
