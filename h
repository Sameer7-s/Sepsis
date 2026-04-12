[33mcommit 326a3e7ff6019c4456ae5b6e18a9c2e7cb8f5cea[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mmain[m[33m, [m[1;31mhf/main[m[33m, [m[1;31mhf/HEAD[m[33m, [m[1;31mgithub/main[m[33m, [m[1;31mgithub/HEAD[m[33m)[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 15:00:34 2026 +0530

    chore: update backend and inference

[33mcommit 1cc8f31fb5366b2a502ca1d2373bb05e63f83680[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 14:25:33 2026 +0530

    fix: clean README frontmatter

[33mcommit 56194aa63980471ebddd55e548235c8779ac3f35[m
Merge: 8fb5984 01cf7c9
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 14:24:29 2026 +0530

    merge github/main into hf-updated main
    
    # Conflicts:
    #       README.md

[33mcommit 8fb59846de2ce05262d2e69b36660ceb3c04ebe0[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 14:21:10 2026 +0530

    fix: resolve merge conflicts

[33mcommit 01cf7c95dae1f51652cc66134a8d7888b8b5bea3[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 14:21:10 2026 +0530

    fix: resolve merge conflicts

[33mcommit b845ad2b2c6634709293b1e0f3172545348c252f[m
Author: s788 <Sam795@users.noreply.huggingface.co>
Date:   Sun Apr 12 08:38:48 2026 +0000

    Update README.md

[33mcommit 74ba994ad5ad0adf91b783afbe369e64b111ac3d[m
Merge: 7e513af 9597bd1
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 13:53:20 2026 +0530

    merge hf remote

[33mcommit 7e513afae8010753bf00cfa99bd9040420c37b07[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 13:50:21 2026 +0530

    updated files

[33mcommit 9597bd10ea6cec1d3e0abfd32314243a9622963c[m
Author: s788 <Sam795@users.noreply.huggingface.co>
Date:   Sun Apr 12 08:06:57 2026 +0000

    Upload 6 files

[33mcommit 55baa0e52ca9cb7aed7c5f38564913fd55a3c2be[m
Author: s788 <Sam795@users.noreply.huggingface.co>
Date:   Sun Apr 12 08:05:20 2026 +0000

    Upload 10 files

[33mcommit 29b834aee0750f294b70b822e136b7096cc8563c[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 13:31:06 2026 +0530

    updated files

[33mcommit 7f52a3beb173f50fca688daca064274360643c74[m
Author: s788 <Sam795@users.noreply.huggingface.co>
Date:   Sun Apr 12 07:52:58 2026 +0000

    initial commit

[33mcommit b5aa3bc39ef06738f829f0ee84defd51e4d2f5d7[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 13:10:11 2026 +0530

    debug: add critical diagnostic probe + sys import + explicit stdout flushing for Phase 2 stdout capture issue

[33mcommit 662e3831fa96d72b398edf1b1b521944e586ad36[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 12:41:25 2026 +0530

    fix: guarantee structured output in all code paths - ensure steps >= 1, always emit [STEP], emit fallback output when backend unavailable

[33mcommit 703f723551325c79d6503931bc3ebe425f40f6c3[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 12:34:43 2026 +0530

    fix: add step_emitted flag to guarantee at least one [STEP] is emitted per task

[33mcommit 03dab814f8776e5c916c825ad013dae3f7282c3c[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 12:28:21 2026 +0530

    refactor: simplify inference.py for cleaner, validator-safe execution

[33mcommit 8dc6e6cefda810a7b5d3eeb7aaedb7ee9544f7a5[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 12:20:53 2026 +0530

    feat: add strictly parseable structured output with emit_start/step/end helpers

[33mcommit 76f66701c97d25702c27eab2040846647aa87950[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 12:14:40 2026 +0530

    fix: ensure importability for OpenEnv validator (add __init__.py at root)

[33mcommit 4d75743cd9151fb90b9792bfdeda1418789374ec[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 11:29:13 2026 +0530

    fix: critical issues - add OpenAI import, fix normalize_severity function, remove unnecessary uvicorn import from backend_api

[33mcommit f03ee69b21aeb011807c38e680cd74419cc24404[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 11:21:44 2026 +0530

    fix: update openenv.yaml entrypoints to match current server/inference architecture

[33mcommit 66459c2c234ec023eda0dbfd37acef2ce5d83411[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 11:11:36 2026 +0530

    refactor: simplify server/app.py - use string-form uvicorn.run() to eliminate double-import issues

[33mcommit 68969e653b829e6bacb1148d05f2d0952ec9fe1b[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 11:05:42 2026 +0530

    fix: Remove port cleanup logic that causes double-binding in containerized environments - simplify server/app.py and remove unused run_server() from backend_api

[33mcommit c4645f4a2a680c1f1f12f5094aae1548b33aa0b7[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 10:51:36 2026 +0530

    fix: add robust port binding retry logic to handle containerized environment delays

[33mcommit 7507744144d54134c3ad1f219a55d1900aa2110f[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 10:40:31 2026 +0530

    fix: clean up finally block in inference.py - remove unnecessary backend process cleanup

[33mcommit 1161a610929f5ec46e83c43a34acd43e2f040043[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 03:23:38 2026 +0530

    fix: /reset endpoint bulletproof for OpenEnv validator - handles empty/invalid JSON gracefully

[33mcommit 375e2496bff87da4eedb99ed39df20af587e7369[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 03:18:04 2026 +0530

    fix: /reset endpoint accepts empty or invalid JSON body for OpenEnv compliance

[33mcommit 7cd28df38cbd8a499a4dab63ed1a949491b67b51[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 03:09:15 2026 +0530

    Fix port binding error: add socket reuse, process cleanup, signal handling, and FastAPI lifecycle events

[33mcommit c8032f0e5d86b2174706bd4d0a4371868dd338da[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 03:07:34 2026 +0530

    Add comprehensive submission validation - final payload, response validation, LLM check

[33mcommit 00f4be31619b6e8d6f37e89dee69b2306c1297e6[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 02:55:32 2026 +0530

    Add comprehensive validation for all conditions - backend and inference

[33mcommit 58ee4d3f78b99d044458f7aa73534c78bdf03427[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 02:51:33 2026 +0530

    updated files

[33mcommit c1654ae6cc2a4b92b721602b97b023ad3c924db9[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 01:35:07 2026 +0530

    updated files

[33mcommit f9f2c9a87044f760750ef5300c4a074775fe5526[m
Author: Sameer-s7 <sameergoutam794@gmail.com>
Date:   Sun Apr 12 01:04:04 2026 +0530

    updated files
