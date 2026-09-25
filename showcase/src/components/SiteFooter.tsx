const LINKS = [
  { href: "https://github.com/nikhilp0799/eqd-risk-engine", label: "Source" },
  { href: "https://github.com/nikhilp0799", label: "GitHub" },
  { href: "https://www.linkedin.com/in/nikhilpandey19", label: "LinkedIn" },
];

export function SiteFooter() {
  return (
    <footer className="mt-20 border-t border-border bg-surface">
      <div className="mx-auto flex max-w-6xl flex-col items-start justify-between gap-3 px-6 py-8 text-sm text-ink-2 sm:flex-row sm:items-center">
        <p>
          EQD Risk Engine, by <span className="font-medium text-ink">Nikhil Pandey</span>
        </p>
        <div className="flex gap-5">
          {LINKS.map((link) => (
            <a
              key={link.label}
              href={link.href}
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-brand"
            >
              {link.label}
            </a>
          ))}
        </div>
      </div>
    </footer>
  );
}
