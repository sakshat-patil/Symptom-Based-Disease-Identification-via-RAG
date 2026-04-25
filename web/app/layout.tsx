import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Record-Based Medical Diagnostic Assistant",
  description:
    "Hybrid FP-Growth + Retrieval-Augmented Pipeline · CMPE 255 · Patil, Kumar, Madhave",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" data-theme="light">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&family=Fraunces:ital,wght@0,500;1,500;1,600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
