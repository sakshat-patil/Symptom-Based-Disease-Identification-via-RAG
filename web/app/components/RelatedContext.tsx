"use client";
import { RelatedContext } from "../api";

// Crude tier inference for the related list (server doesn't echo tier).
function tierFromSource(source: string): number {
  const tier1 = [
    "1_CancerGov_QA",
    "2_GARD_QA",
    "3_GHR_QA",
    "4_MPlus_Health_Topics_QA",
    "5_NIDDK_QA",
    "6_NINDS_QA",
    "8_NHLBI_QA_XML",
    "9_CDC_QA",
  ];
  const tier2 = ["7_SeniorHealth_QA", "10_MPlus_ADAM_QA", "11_MPlusDrugs_QA"];
  if (tier1.includes(source)) return 1;
  if (tier2.includes(source)) return 2;
  return 3;
}

function snippet(text: string, maxWords = 28) {
  const cleaned = text.replace(/^Q:.*?A:\s*/s, "").replace(/\s+/g, " ").trim();
  const words = cleaned.split(" ");
  return words.slice(0, maxWords).join(" ") + (words.length > maxWords ? "…" : "");
}

export function RelatedContextPanel({ items }: { items: RelatedContext[] }) {
  if (!items.length) return null;
  return (
    <>
      <h3 className="h-section">
        Related biomedical context
        <span>{items.length} nearby passages</span>
      </h3>
      <div className="card" style={{ padding: "4px 16px" }}>
        {items.map((it) => {
          const tier = tierFromSource(it.source);
          return (
            <div key={it.passage_id} className="related-row">
              <span className="related-row__src">
                {it.source_label}
                <span
                  className={`pill pill--tier${tier}`}
                  style={{ marginLeft: 6, fontSize: 10 }}
                >
                  <span className="dot" />t{tier}
                </span>
              </span>
              <span className="pill" style={{ justifySelf: "start" }}>
                passage
              </span>
              <span className="related-row__text">
                <strong style={{ color: "var(--ink)", fontWeight: 600 }}>
                  {it.focus}.
                </strong>{" "}
                {snippet(it.text)}
              </span>
              <span className="related-row__sim">
                sim {it.score.toFixed(3)}
              </span>
            </div>
          );
        })}
      </div>
    </>
  );
}
