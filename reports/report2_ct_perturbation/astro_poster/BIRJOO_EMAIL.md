# Draft cover email to Birjoo — ASTRO #79011 poster (DRAFT, Neil sends)

Attach: `ASTRO_79011_poster_draft.pdf` (2-page draft: page 1 at-a-glance, page 2 the figures).

---

**Subject:** ASTRO #79011 poster draft — two quick questions before I finalize

Hi Birjoo,

Attached is a draft of the ASTRO poster (#79011) built from the finished CT-perturbation sweep
(40 patients, 26 conditions). It's content + figures for review, not the final laid-out kiosk
version yet.

The single-glance result holds up cleanly: **four of five perturbation families are robust; spatial
resolution is the one that bites.** Under a "clinically visible = any DVH criterion shifts >1 Gy"
criterion, P4/resolution crosses at a realistic severity (L2, ~2 mm-equivalent blur; Larynx D0.1cc
1.15 Gy → 2.84 Gy at L4), while noise, HU shift, bias field, and dental streaks never cross 1 Gy in
range. Practical read: cross-site QA should prioritize resolution / reconstruction-kernel
consistency over intensity calibration.

Two things I'd like your call on, since you're presenting:

1. **The threshold.** I set "clinically visible" at **>1 Gy on any DVH criterion** (vs the % framing
   in the earlier report). It's sharper and the data supports it — but is 1 Gy the line you want to
   stand behind clinically?
2. **The "vs. clinical range" comparisons** on the table (e.g. calibration drift ≈10–50 HU,
   slice/kernel variation ≈1–3 voxels) — you'll know better than me if those ranges are right to
   cite.

One more in progress: the interactive element (a GIF stepping one patient through the resolution
sweep) — script's ready, I'll add it this week.

Happy to adjust anything. Deadline for upload is Sept 21.

Thanks,
Neil
