# Laya test results

Environment: laya 0.3.22, torch 2.14.0, Python 3.12.9, arm64, MPS=True. Latency is the median of 5 warm runs per case (all questions for a case in one call).

**Accuracy: 26/28 questions (93%)**

| Case | Question | Expected | Predicted | Confidence | OK | Model | ms |
|---|---|---|---|---|---|---|---|
| triage: duplicate charge | department | billing | billing | 0.983 | ✅ | english | 45 |
| triage: duplicate charge | urgency | critical deadline or blocking issue | critical deadline or blocking issue (score=1.56) | 0.626 | ✅ | english | 45 |
| triage: duplicate charge | churn_risk | yes | yes (p=0.879) | 0.879 | ✅ | english | 45 |
| triage: duplicate charge | refund_requested | yes | yes (p=0.911) | 0.911 | ✅ | english | 45 |
| triage: app crash | department | technical | technical | 0.874 | ✅ | english | 41 |
| triage: app crash | urgency | critical deadline or blocking issue | critical deadline or blocking issue (score=1.91) | 0.919 | ✅ | english | 41 |
| triage: app crash | churn_risk | no | no (p=0.123) | 0.877 | ✅ | english | 41 |
| triage: app crash | refund_requested | no | no (p=0.039) | 0.961 | ✅ | english | 41 |
| triage: pricing enquiry | department | sales | sales | 0.715 | ✅ | english | 43 |
| triage: pricing enquiry | urgency | not urgent | not urgent (score=0.30) | 0.766 | ✅ | english | 43 |
| triage: pricing enquiry | churn_risk | no | no (p=0.009) | 0.991 | ✅ | english | 43 |
| triage: JSON email state | department | billing | billing | 0.962 | ✅ | english | 55 |
| triage: JSON email state | churn_risk | yes | yes (p=0.816) | 0.816 | ✅ | english | 55 |
| triage: JSON email state | refund_requested | yes | yes (p=0.846) | 0.846 | ✅ | english | 55 |
| guardrail: prompt injection | injection | yes | yes (p=0.927) | 0.927 | ✅ | english | 33 |
| guardrail: prompt injection | toxic | no | no (p=0.083) | 0.917 | ✅ | english | 33 |
| guardrail: benign question | injection | no | no (p=0.000) | 1.000 | ✅ | english | 34 |
| guardrail: benign question | toxic | no | no (p=0.075) | 0.924 | ✅ | english | 34 |
| guardrail: abusive | injection | no | no (p=0.000) | 1.000 | ✅ | english | 31 |
| guardrail: abusive | toxic | yes | yes (p=0.822) | 0.822 | ✅ | english | 31 |
| sentiment: HF widget example | sentiment | positive | positive | 0.617 | ✅ | english | 27 |
| sentiment: negative | sentiment | negative | negative | 0.977 | ✅ | english | 24 |
| multilingual: Hindi billing | department | billing | billing | 0.645 | ✅ | multilingual | 23 |
| multilingual: Hindi billing | refund_requested | yes | yes (p=0.981) | 0.981 | ✅ | multilingual | 23 |
| multilingual: Spanish crash | department | technical | technical | 0.996 | ✅ | multilingual | 21 |
| multilingual: French cancel | churn_risk | yes | no (p=0.062) | 0.938 | ❌ | multilingual | 20 |
| multilingual: Japanese pricing | department | sales | billing | 0.999 | ❌ | multilingual | 16 |
| multilingual: German abusive | toxic | yes | yes (p=0.783) | 0.783 | ✅ | multilingual | 16 |
