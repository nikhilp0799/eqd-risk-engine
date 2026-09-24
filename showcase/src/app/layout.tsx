import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { SiteNav } from "@/components/SiteNav";
import { SiteFooter } from "@/components/SiteFooter";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "EQD Risk Engine — Quant Derivatives Risk Platform",
  description:
    "A from-scratch equity derivatives risk engine: vol surface calibration, Monte Carlo exotics pricing, deep-hedging neural policies, and an agentic AI investigator — validated against real market data and QuantLib.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <SiteNav />
        <main className="flex-1 mx-auto w-full max-w-6xl px-6 py-10">
          {children}
        </main>
        <SiteFooter />
      </body>
    </html>
  );
}
