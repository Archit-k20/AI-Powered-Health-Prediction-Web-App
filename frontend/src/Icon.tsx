export function Icon({
  name,
  className = "",
}: {
  name:
    "symptoms" | "report" | "summary" | "shield" | "arrow" | "check" | "heart";
  className?: string;
}) {
  const paths = {
    symptoms: (
      <>
        <path d="M5 3v5a4 4 0 0 0 8 0V3M3 3h4m4 0h4M9 12v3a5 5 0 0 0 10 0v-2" />
        <circle cx="19" cy="10" r="3" />
      </>
    ),
    report: (
      <>
        <path d="M14 3H5v18h14V8zM14 3v5h5M8 12h8M8 16h5" />
      </>
    ),
    summary: (
      <>
        <rect x="4" y="5" width="16" height="16" rx="3" />
        <path d="M9 5V3h6v2M8 12l2 2 5-5M8 18h8" />
      </>
    ),
    shield: (
      <>
        <path d="M12 3 4 6v6c0 5 8 9 8 9s8-4 8-9V6zM8 12l3 3 5-6" />
      </>
    ),
    arrow: <path d="M4 12h15m-6-6 6 6-6 6" />,
    check: <path d="m5 12 4 4L19 6" />,
    heart: (
      <>
        <path d="M20 5c-3-3-7 0-8 2-1-2-5-5-8-2-4 4 1 9 8 15 7-6 12-11 8-15Z" />
        <path d="M3 12h5l2-4 3 8 2-4h6" />
      </>
    ),
  };
  return (
    <svg
      className={`icon ${className}`}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}
