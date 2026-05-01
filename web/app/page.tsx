"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import {
  motion,
  useInView,
  useScroll,
  useTransform,
  type Variants,
} from "framer-motion";
import { useAuth } from "./components/AuthProvider";
import { UserMenu } from "./components/UserMenu";

/**
 * Public landing page. Calm-but-polished AI-research-lab voice. Animation
 * budget is intentionally restrained: a hero word-stagger, on-view stat
 * counters, scroll-revealed sections, and hover lift on feature cards.
 * No constant motion noise; one moment of magic per fold.
 */
export default function Landing() {
  const { user, loading } = useAuth();
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  return (
    <div className="landing">
      <LandingTopbar
        theme={theme}
        setTheme={setTheme}
        user={user}
        loading={loading}
      />

      <Hero authed={Boolean(user)} />

      <MetricsRow />

      <HowItWorks />

      <Auditable />

      <FinalCta authed={Boolean(user)} />

      <LandingFooter />
    </div>
  );
}

/* =================================================================
   TOPBAR
   ================================================================= */

function LandingTopbar({
  theme,
  setTheme,
  user,
  loading,
}: {
  theme: "light" | "dark";
  setTheme: (t: "light" | "dark") => void;
  user: ReturnType<typeof useAuth>["user"];
  loading: boolean;
}) {
  return (
    <motion.header
      className="landing__topbar"
      initial={{ y: -8, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.4, ease: [0.2, 0.7, 0.2, 1] }}
    >
      <div className="landing__brand">
        <div className="topbar__logo">R</div>
        <div>
          <div className="topbar__title">
            Record-Based Medical Diagnostic Assistant
          </div>
          <div className="topbar__sub">
            CMPE 255, San Jose State University
          </div>
        </div>
      </div>
      <div style={{ flex: 1 }} />
      <button
        className="icon-btn"
        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        title="Toggle theme"
        aria-label="Toggle theme"
      >
        {theme === "dark" ? "☀" : "☾"}
      </button>
      {!loading && user ? (
        <>
          <Link href="/app" className="btn btn--ghost">
            Open app
          </Link>
          <UserMenu />
        </>
      ) : (
        <div className="landing__auth-cta">
          <Link href="/login" className="btn">
            Sign in
          </Link>
          <Link href="/register" className="btn landing__btn-accent">
            Get started
          </Link>
        </div>
      )}
    </motion.header>
  );
}

/* =================================================================
   HERO
   ================================================================= */

const HEAD_TOP = "A symptom-first differential,";
const HEAD_BOT_PRE = "with";
const HEAD_BOT_EM = "citable evidence";
const HEAD_BOT_POST = "behind every prediction.";

function Hero({ authed }: { authed: boolean }) {
  // Subtle parallax on the mesh background as the page scrolls.
  const ref = useRef<HTMLElement | null>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start start", "end start"],
  });
  const meshY = useTransform(scrollYProgress, [0, 1], [0, 80]);
  const meshOpacity = useTransform(scrollYProgress, [0, 1], [1, 0.35]);

  const wordContainer: Variants = {
    hidden: {},
    show: {
      transition: { staggerChildren: 0.045, delayChildren: 0.15 },
    },
  };
  const word: Variants = {
    hidden: { y: "0.6em", opacity: 0, filter: "blur(4px)" },
    show: {
      y: 0,
      opacity: 1,
      filter: "blur(0px)",
      transition: { duration: 0.55, ease: [0.2, 0.7, 0.2, 1] },
    },
  };

  return (
    <section className="landing__hero" ref={ref}>
      <motion.div
        className="landing__hero-mesh"
        style={{ y: meshY, opacity: meshOpacity }}
        aria-hidden
      />
      <div className="landing__hero-grid" aria-hidden />

      <motion.div
        className="landing__hero-eyebrow"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.05 }}
      >
        <span className="landing__pulse" aria-hidden />
        Live research preview · CMPE 255 final project
      </motion.div>

      <motion.h1
        className="landing__hero-title"
        variants={wordContainer}
        initial="hidden"
        animate="show"
      >
        <span className="landing__hero-line">
          {HEAD_TOP.split(" ").map((w, i) => (
            <motion.span key={i} className="landing__word" variants={word}>
              <span>{w}&nbsp;</span>
            </motion.span>
          ))}
        </span>
        <span className="landing__hero-line">
          {HEAD_BOT_PRE.split(" ").map((w, i) => (
            <motion.span
              key={`a${i}`}
              className="landing__word"
              variants={word}
            >
              <span>{w}&nbsp;</span>
            </motion.span>
          ))}
          <motion.span className="landing__word" variants={word}>
            <span className="landing__hero-em">
              <em>{HEAD_BOT_EM}</em>
            </span>
            &nbsp;
          </motion.span>
          {HEAD_BOT_POST.split(" ").map((w, i) => (
            <motion.span
              key={`b${i}`}
              className="landing__word"
              variants={word}
            >
              <span>{w}&nbsp;</span>
            </motion.span>
          ))}
        </span>
      </motion.h1>

      <motion.p
        className="landing__hero-lede"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.55 }}
      >
        Two complementary signals, one ranked answer. Association rules
        mined from real patient records meet a biomedical retrieval
        index built from <strong>24,063 NIH passages</strong>. Every
        diagnosis is auditable through the rule that fired and the
        passage that grounds it.
      </motion.p>

      <motion.div
        className="landing__hero-ctas"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.7 }}
      >
        <Link
          href={authed ? "/app" : "/register"}
          className="btn landing__btn-primary landing__btn-shine"
        >
          {authed ? "Open the diagnostic tool" : "Create a doctor account"}
          <span className="landing__btn-arrow" aria-hidden>
            →
          </span>
        </Link>
        <Link
          href={authed ? "/insights" : "/login"}
          className="btn landing__btn-ghost"
        >
          {authed ? "View insights" : "I already have an account"}
        </Link>
      </motion.div>

      <motion.div
        className="landing__hero-disclaimer"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 1.0 }}
      >
        Research project, CMPE 255 spring 2026. Not a clinical device,
        not for use in patient care.
      </motion.div>

      <motion.div
        className="landing__scroll-cue"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.3, duration: 0.6 }}
        aria-hidden
      >
        <span className="landing__scroll-text">Scroll</span>
        <span className="landing__scroll-line" />
      </motion.div>
    </section>
  );
}

/* =================================================================
   METRICS
   ================================================================= */

type Metric = {
  label: string;
  value: number;
  format: (n: number) => string;
  sub: string;
};

const METRICS: Metric[] = [
  {
    label: "Recall @ 1",
    value: 0.825,
    format: (n) => n.toFixed(3),
    sub: "fused, MiniLM, α = 0.3",
  },
  {
    label: "Recall @ 10",
    value: 0.89,
    format: (n) => n.toFixed(3),
    sub: "+2.0 vs. mining-only",
  },
  {
    label: "MRR",
    value: 0.857,
    format: (n) => n.toFixed(3),
    sub: "200 held-out cases",
  },
  {
    label: "p95 latency",
    value: 20,
    format: (n) => `${Math.round(n)} ms`,
    sub: "M3 Pro, default config",
  },
];

function MetricsRow() {
  const ref = useRef<HTMLDivElement | null>(null);
  const inView = useInView(ref, { once: true, amount: 0.4 });
  return (
    <div className="landing__metrics" ref={ref}>
      {METRICS.map((m, i) => (
        <MetricCard key={m.label} metric={m} active={inView} index={i} />
      ))}
    </div>
  );
}

function MetricCard({
  metric,
  active,
  index,
}: {
  metric: Metric;
  active: boolean;
  index: number;
}) {
  const display = useCountUp(active ? metric.value : 0, {
    durationMs: 1100,
    delayMs: 80 + index * 80,
  });
  return (
    <motion.div
      className="landing__metric"
      initial={{ opacity: 0, y: 14 }}
      animate={active ? { opacity: 1, y: 0 } : { opacity: 0, y: 14 }}
      transition={{
        duration: 0.5,
        delay: 0.05 + index * 0.06,
        ease: [0.2, 0.7, 0.2, 1],
      }}
      whileHover={{ y: -3 }}
    >
      <div className="landing__metric-label">{metric.label}</div>
      <div className="landing__metric-value">
        <span className="landing__metric-num">{metric.format(display)}</span>
      </div>
      <div className="landing__metric-sub">{metric.sub}</div>
    </motion.div>
  );
}

/* =================================================================
   HOW IT WORKS (3 features)
   ================================================================= */

function HowItWorks() {
  const features = [
    {
      num: "01",
      title: "Mined from records",
      body: (
        <>
          FP-Growth over a 4,920-row, 41-disease patient transaction table
          yields <strong>23,839 association rules</strong>. At query time
          we score each rule against the patient&rsquo;s symptom set with
          overlap-weighted confidence.
        </>
      ),
    },
    {
      num: "02",
      title: "Grounded in literature",
      body: (
        <>
          Dense retrieval over <strong>24,063 MedQuAD passages</strong>{" "}
          (NIH MedlinePlus, NHLBI, NIDDK, GARD). Three encoder backends:
          MiniLM 384d, PubMedBERT 768d, Azure OpenAI 3072d. Curated
          clinical synonym bridge.
        </>
      ),
    },
    {
      num: "03",
      title: "Fused, not stacked",
      body: (
        <>
          <span className="mono">
            FusedScore(d) = α · RetrievalSim(d) + (1 − α) · MiningConf(d)
          </span>
          . Default α = 0.3 sits inside the optimal plateau. Drag the
          slider live in the app and watch rankings re-fuse.
        </>
      ),
    },
  ];

  return (
    <section className="landing__section">
      <SectionHead
        eyebrow="How it works"
        title="Two signals. One ranked differential."
      />
      <div className="landing__features">
        {features.map((f, i) => (
          <FeatureCard key={f.num} feature={f} index={i} />
        ))}
      </div>
    </section>
  );
}

function FeatureCard({
  feature,
  index,
}: {
  feature: { num: string; title: string; body: React.ReactNode };
  index: number;
}) {
  const ref = useRef<HTMLElement | null>(null);
  const inView = useInView(ref, { once: true, amount: 0.35 });
  return (
    <motion.article
      ref={ref}
      className="landing__feature"
      initial={{ opacity: 0, y: 24 }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 24 }}
      transition={{
        duration: 0.55,
        delay: index * 0.08,
        ease: [0.2, 0.7, 0.2, 1],
      }}
      whileHover={{ y: -4 }}
    >
      <div className="landing__feature-glow" aria-hidden />
      <div className="landing__feature-num">{feature.num}</div>
      <h3 className="landing__feature-title">{feature.title}</h3>
      <p className="landing__feature-body">{feature.body}</p>
    </motion.article>
  );
}

/* =================================================================
   AUDITABLE BY CONSTRUCTION
   ================================================================= */

type AuditCard = {
  icon: React.ReactNode;
  title: string;
  body: React.ReactNode;
  hint: string;
};

function Auditable() {
  const cards: AuditCard[] = [
    {
      icon: <IconHighlight />,
      title: "Claim-level evidence cards",
      body: (
        <>
          The exact sentence that mentions the symptom or disease is{" "}
          <span className="landing__hl">highlighted</span>, with character
          offsets, a source-authority tier, and a passage-type label.
        </>
      ),
      hint: "tier 1 · MedlinePlus, NHLBI, NIDDK",
    },
    {
      icon: <IconStructured />,
      title: "Structured clinical explanation",
      body: (
        <>
          Four sections per ranked diagnosis: symptom-disease link,
          statistical prior, evidence quality, what is missing. Citations
          appended <em>deterministically</em>, so LLM prose stays traceable.
        </>
      ),
      hint: "src/clinical_explanation.py",
    },
    {
      icon: <IconTimeline />,
      title: "Live pipeline timeline",
      body: (
        <>
          Watch every stage execute. Encode, retrieve, attribute, mine,
          fuse, evidence, explain. Real per-stage latencies snap in the
          moment the response lands.
        </>
      ),
      hint: "p50 7.6 ms · p95 19.6 ms",
    },
    {
      icon: <IconLayers />,
      title: "Two operating modes",
      body: (
        <>
          Offline path: FAISS + template explainer, no keys, 12 ms p50.
          Production path: Azure OpenAI 3072d on Pinecone Serverless,
          GPT-5.3 explainer.
        </>
      ),
      hint: "VECTOR_STORE = faiss | pinecone",
    },
  ];

  return (
    <section className="landing__section landing__section--alt">
      <SectionHead
        eyebrow="Auditable by construction"
        title="Every claim cites a passage. Every passage cites a source."
      />
      <div className="audit-grid">
        {cards.map((c, i) => (
          <AuditCardEl key={c.title} card={c} index={i} />
        ))}
      </div>
    </section>
  );
}

function AuditCardEl({ card, index }: { card: AuditCard; index: number }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const inView = useInView(ref, { once: true, amount: 0.35 });
  return (
    <motion.div
      ref={ref}
      className="audit-card"
      initial={{ opacity: 0, y: 18 }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 18 }}
      transition={{
        duration: 0.5,
        delay: index * 0.07,
        ease: [0.2, 0.7, 0.2, 1],
      }}
      whileHover={{ y: -3 }}
    >
      <div className="audit-card__icon" aria-hidden>
        {card.icon}
      </div>
      <h3 className="audit-card__title">{card.title}</h3>
      <p className="audit-card__body">{card.body}</p>
      <div className="audit-card__hint">
        <span className="audit-card__hint-dot" aria-hidden />
        <span>{card.hint}</span>
      </div>
    </motion.div>
  );
}

/* Small, monoline SVG icons for the audit cards. ~18 px, currentColor. */

function IconHighlight() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M5 8h14" />
      <rect x="5" y="11" width="9" height="3" rx="0.5" fill="currentColor" stroke="none" opacity="0.18" />
      <path d="M5 11h9" />
      <path d="M5 14h14" />
      <path d="M5 17h10" />
    </svg>
  );
}

function IconStructured() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="4" y="4" width="7" height="7" rx="1.2" />
      <rect x="13" y="4" width="7" height="7" rx="1.2" />
      <rect x="4" y="13" width="7" height="7" rx="1.2" />
      <rect x="13" y="13" width="7" height="7" rx="1.2" />
    </svg>
  );
}

function IconTimeline() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="6" cy="12" r="2" fill="currentColor" stroke="none" />
      <circle cx="12" cy="12" r="2" />
      <circle cx="18" cy="12" r="2" />
      <path d="M8 12h2" />
      <path d="M14 12h2" />
    </svg>
  );
}

function IconLayers() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 4l8 4-8 4-8-4 8-4z" />
      <path d="M4 12l8 4 8-4" />
      <path d="M4 16l8 4 8-4" />
    </svg>
  );
}

/* =================================================================
   FINAL CTA
   ================================================================= */

function FinalCta({ authed }: { authed: boolean }) {
  const ref = useRef<HTMLElement | null>(null);
  const inView = useInView(ref, { once: true, amount: 0.4 });
  return (
    <motion.section
      ref={ref}
      className="landing__cta"
      initial={{ opacity: 0, y: 20 }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
      transition={{ duration: 0.55, ease: [0.2, 0.7, 0.2, 1] }}
    >
      <div className="landing__cta-inner">
        <h2 className="landing__cta-title">
          Sign in and try the cardiac event preset.
        </h2>
        <p className="landing__cta-body">
          The fastest way to understand the system is to run it. Pick the
          preset, watch Heart Attack rank #1 with fused score 0.831, and
          drag the α slider to see the lift collapse without mining.
        </p>
        <div className="landing__hero-ctas">
          {authed ? (
            <Link
              href="/app"
              className="btn landing__btn-primary landing__btn-shine"
            >
              Open the app
              <span className="landing__btn-arrow" aria-hidden>
                →
              </span>
            </Link>
          ) : (
            <>
              <Link
                href="/register"
                className="btn landing__btn-primary landing__btn-shine"
              >
                Create account
                <span className="landing__btn-arrow" aria-hidden>
                  →
                </span>
              </Link>
              <Link href="/login" className="btn landing__btn-ghost">
                Sign in
              </Link>
            </>
          )}
        </div>
      </div>
    </motion.section>
  );
}

/* =================================================================
   FOOTER
   ================================================================= */

function LandingFooter() {
  return (
    <footer className="landing__footer">
      <div className="landing__footer-row">
        <div className="landing__footer-brand">
          <div className="topbar__logo">R</div>
          <div>
            <div className="landing__footer-title">
              Record-Based Medical Diagnostic Assistant
            </div>
            <div className="landing__footer-sub">
              Final project, CMPE 255 Data Mining, spring 2026
            </div>
          </div>
        </div>
        <div className="landing__footer-meta">
          <div>Sakshat Patil · Vineet Kumar · Aishwarya Madhave</div>
          <div className="landing__footer-sub">
            San Jose State University, Department of Computer Engineering
          </div>
        </div>
      </div>
      <div className="landing__footer-fineprint">
        Research project. Numbers shown reflect 200 synthetic test cases,
        not a clinical evaluation. Not for use in patient care.
      </div>
    </footer>
  );
}

/* =================================================================
   SHARED PIECES
   ================================================================= */

function SectionHead({
  eyebrow,
  title,
}: {
  eyebrow: string;
  title: string;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const inView = useInView(ref, { once: true, amount: 0.5 });
  return (
    <motion.div
      ref={ref}
      className="landing__section-head"
      initial={{ opacity: 0, y: 14 }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 14 }}
      transition={{ duration: 0.5 }}
    >
      <div className="landing__section-eyebrow">{eyebrow}</div>
      <h2 className="landing__section-title">{title}</h2>
    </motion.div>
  );
}

/* =================================================================
   useCountUp: gentle ease-out tween from 0 to target.
   Used by metric cards once they enter the viewport.
   ================================================================= */

function useCountUp(
  target: number,
  opts: { durationMs?: number; delayMs?: number } = {},
): number {
  const { durationMs = 900, delayMs = 0 } = opts;
  const [value, setValue] = useState(0);

  useEffect(() => {
    if (target === 0) {
      setValue(0);
      return;
    }
    let raf = 0;
    let start = 0;
    const t0 = performance.now() + delayMs;
    const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3);
    const tick = (now: number) => {
      if (now < t0) {
        raf = requestAnimationFrame(tick);
        return;
      }
      if (!start) start = now;
      const elapsed = now - t0;
      const t = Math.min(1, elapsed / durationMs);
      setValue(target * easeOutCubic(t));
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, durationMs, delayMs]);

  return value;
}
