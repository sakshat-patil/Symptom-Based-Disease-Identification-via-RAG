"use client";

type IconName =
  | "x" | "search" | "chevron" | "chevron-r" | "chevron-d"
  | "filter" | "sliders" | "info" | "warn"
  | "link" | "scale" | "check" | "moon" | "sun" | "sparkle";

export function Icon({ name, size = 14 }: { name: IconName; size?: number }) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
  };
  switch (name) {
    case "x":
      return <svg {...common}><path d="M6 6l12 12M6 18L18 6" /></svg>;
    case "search":
      return <svg {...common}><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></svg>;
    case "chevron":
      return <svg {...common}><path d="m6 9 6 6 6-6" /></svg>;
    case "chevron-r":
      return <svg {...common}><path d="m9 6 6 6-6 6" /></svg>;
    case "chevron-d":
      return <svg {...common}><path d="m6 9 6 6 6-6" /></svg>;
    case "filter":
      return <svg {...common}><path d="M3 5h18M6 12h12M10 19h4" /></svg>;
    case "sliders":
      return (
        <svg {...common}>
          <path d="M4 6h12M20 6h0M4 12h4M12 12h8M4 18h12M20 18h0" />
          <circle cx="18" cy="6" r="2" />
          <circle cx="10" cy="12" r="2" />
          <circle cx="18" cy="18" r="2" />
        </svg>
      );
    case "info":
      return <svg {...common}><circle cx="12" cy="12" r="9" /><path d="M12 8v0M12 11v5" /></svg>;
    case "warn":
      return (
        <svg {...common}>
          <path d="M10.3 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
          <path d="M12 9v4M12 17h0" />
        </svg>
      );
    case "link":
      return (
        <svg {...common}>
          <path d="M10 13a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-1 1" />
          <path d="M14 11a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l1-1" />
        </svg>
      );
    case "scale":
      return <svg {...common}><path d="M3 6h18M7 6v3a5 5 0 0 0 10 0V6M12 13v8" /></svg>;
    case "check":
      return <svg {...common}><path d="m4 12 5 5L20 6" /></svg>;
    case "moon":
      return <svg {...common}><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" /></svg>;
    case "sparkle":
      return (
        <svg {...common}>
          <path d="M12 3l1.8 4.5L18 9l-4.2 1.5L12 15l-1.8-4.5L6 9l4.2-1.5L12 3z" />
          <path d="M19 14l.9 2.1L22 17l-2.1.9L19 20l-.9-2.1L16 17l2.1-.9L19 14z" />
        </svg>
      );
    case "sun":
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
        </svg>
      );
    default:
      return null;
  }
}
