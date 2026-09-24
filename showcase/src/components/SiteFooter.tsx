export function SiteFooter() {
  return (
    <footer className="border-t border-panel-border mt-16">
      <div className="mx-auto max-w-6xl px-6 py-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-sm text-muted">
        <p>
          Built by <span className="text-foreground">Nikhil Pandey</span> — quant risk
          engineering &amp; derivatives pricing.
        </p>
        <div className="flex gap-4 font-num">
          {/* TODO: replace with real profile URLs before deploying */}
          <a href="#" className="hover:text-accent transition-colors">
            GitHub
          </a>
          <a href="#" className="hover:text-accent transition-colors">
            LinkedIn
          </a>
        </div>
      </div>
    </footer>
  );
}
