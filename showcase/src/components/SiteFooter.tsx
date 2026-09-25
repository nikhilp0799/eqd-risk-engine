const LINKS = [
  { href: "https://github.com/nikhilp0799/eqd-risk-engine", label: "Source" },
  { href: "https://github.com/nikhilp0799", label: "GitHub" },
  { href: "https://www.linkedin.com/in/nikhilpandey19", label: "LinkedIn" },
];

export function SiteFooter() {
  return (
    <footer className="border-t border-panel-border mt-16">
      <div className="mx-auto max-w-6xl px-6 py-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-sm text-muted">
        <p>
          Built by <span className="text-foreground">Nikhil Pandey</span> — quant risk
          engineering &amp; derivatives pricing.
        </p>
        <div className="flex gap-4 font-num">
          {LINKS.map((link) => (
            <a
              key={link.label}
              href={link.href}
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-accent transition-colors"
            >
              {link.label}
            </a>
          ))}
        </div>
      </div>
    </footer>
  );
}
