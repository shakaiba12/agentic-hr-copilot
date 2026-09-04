"""
Query Orchestration Gate & Routing Layer for Enterprise RAG Pipeline (Step 1).
Deterministic, fail-closed classification across safety boundaries and target handlers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Union

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from src.guardrails.input_guardrail import InputGuardrail, InputGuardrailResult


class RouteCategory(str, Enum):
    """Explicit routing categories for input orchestration."""

    GENERAL = "GENERAL"
    CASUAL = "CASUAL"
    RAG_KNOWLEDGE = "RAG_KNOWLEDGE"
    DATA_QUERY = "DATA_QUERY"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    INVALID = "INVALID"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    DESTRUCTIVE_ACTION = "DESTRUCTIVE_ACTION"
    SECRET_REQUEST = "SECRET_REQUEST"
    SYSTEM_MANIPULATION = "SYSTEM_MANIPULATION"


@dataclass(frozen=True)
class RouteDecision:
    """Structured decision output from the query orchestration gate."""

    category: RouteCategory
    allowed: bool
    target: Optional[str]  # "rag", "data_specialist", "general", None
    reason: str
    confidence: Optional[float] = 1.0


class QueryRouter:
    """
    Deterministic Query Router and Orchestration Gate.
    Routes queries to RAG, Data Specialist, General Handler, or blocks them (Fail-Closed).
    """

    # Pure greetings and conversational pleasantries (strictly anchored to prevent stealing real queries)
    _CASUAL_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"(?i)^(hi|hello|hey|heya|hiya|hy|howdy|sup|wassup|wazzup|yo|hola|greetings|good\s+(morning|afternoon|evening|day)|thanks|thank\s+you|bye|goodbye|see\s+you)[\s.!,?]*$"),
        re.compile(r"(?i)^(hi|hello|hey|heya|hiya|hy)\s+(there|assistant|bot)?[\s.!,?]*$"),
        re.compile(r"(?i)^(hi|hello|hey|heya|hy|good\s+(morning|afternoon|evening))[\s,!.-]+(how\s+are\s+you|how\s+can\s+you\s+help|what\s+can\s+you\s+do|what\s+can\s+you\s+help).*$"),
        re.compile(r"(?i)^how\s+are\s+you(\s+doing)?[\s.!,?]*$"),
        re.compile(r"(?i)^(how|what)\s+can\s+you\s+help(\s+me)?(\s+with|\s+today)?[\s.!,?]*$"),
        re.compile(r"(?i)^what\s+(can|do)\s+you\s+do[\s.!,?]*$"),
        re.compile(r"(?i)^what('?s|\s+is)\s+up[\s.!,?]*$"),
        re.compile(r"(?i)^nice\s+to\s+meet\s+you[\s.!,?]*$"),
        re.compile(r"(?i)^help(\s+me)?[\s.!,?]*$"),
    )

    # General / World knowledge questions answered without RAG or SQL
    _GENERAL_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"(?i)\b(what\s+is|explain(\s+what\s+is)?)\s+(python|fastapi|docker|kubernetes|javascript|html|css|linux|sql|git|react|c\+\+|rust|java)\b"),
        re.compile(r"(?i)\b(explain\s+what\s+|explain\s+|what\s+is\s+)sql(\s+is)?\b"),
        re.compile(r"(?i)\btell\s+me\s+a\s+joke(\s+about\s+.*)?\b"),
        re.compile(r"(?i)\bwho\s+(was|is)\s+(albert\s+einstein|isaac\s+newton|william\s+shakespeare|elon\s+musk|marie\s+curie|alan\s+turing|ada\s+lovelace)\b"),
        re.compile(r"(?i)\bwhat\s+is\s+(\d+\s*[\+\-\*\/]\s*\d+|the\s+capital\s+of\s+[a-zA-Z\s]+)\b"),
        re.compile(r"(?i)\b(capital\s+of\s+(france|germany|spain|japan|canada|italy|australia)|weather\s+today|meaning\s+of\s+life)\b"),
        re.compile(r"(?i)\bwhat\s+does\s+['\"].*?['\"]\s+mean\b"),
    )

    # Out of scope inquiries
    _OUT_OF_SCOPE_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"(?i)\b(what|who|tell\s+me\s+about)\s+(is\s+)?wamo\b"),
        re.compile(r"(?i)\bwhat\s+does\s+wamo\s+do\b"),
    )

    # Operational database inquiries (strictly requiring live records, aggregates, statistics, counts)
    _DATA_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = (
        # Employee email / identity lookups
        re.compile(r"(?i)\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b.*?\b(mail\s+of|email\s+of|employee|name|who|which|find|owner|user|belongs|profile|details|person)\b"),
        re.compile(r"(?i)\b(mail\s+of|email\s+of|who\s+has|who\s+uses|who\s+is|who\s+owns|which\s+employee|what\s+employee|find\s+(the\s+)?employee|name\s+of|employee\s+name|contact\s+of|profile\s+for|lookup|look\s+up|search\s+for|owner\s+of)\b.*?\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"),
        re.compile(r"(?i)\bwho\s+is\s+[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"),
        re.compile(r"(?i)\b(what|which)\s+employee\s+(has|uses|is)\s+[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"),
        re.compile(r"(?i)\b(what\s+is|find|get|show)\s+[a-zA-Z]+(\s+[a-zA-Z]+)?'?s?\s+(email|e-mail|mail\s+address|email\s+address)\b"),
        re.compile(r"(?i)\b(email|e-mail)\s+(of|for)\s+[a-zA-Z]+(\s+[a-zA-Z]+)?\b"),
        re.compile(r"(?i)\bwhat\s+(email|e-mail)\s+does\s+[a-zA-Z]+(\s+[a-zA-Z]+)?\s+(use|have)\b"),
        re.compile(r"(?i)\bemployee\s+(email|mail|name|contact)\s+(for|of)\s+[a-zA-Z]+(\s+[a-zA-Z]+)?\b"),
        re.compile(r"(?i)\b(what\s+is\s+(the\s+)?|what\s+|get\s+(the\s+)?|give\s+me\s+(the\s+)?|total\s+|current\s+)?(count|number|total|percentage|percent|proportion|share|ratio)\s+of\s+(current\s+|curent\s+|all\s+|total\s+)?(employees?|employes?|employess?|staff|people|workers?|users?|doctors?|patients?|records?|positions?|departments?)\b"),
        re.compile(r"(?i)\b(\w+\s+)?(employees?|employe|employess|staff|worker)\s+(count|total|number|records?|database|information|salaries|salary|sary|salery|pay|compensation)\b"),
        re.compile(r"(?i)\b(salary|salaries|sary|salery|pay|compensation)\s+of\s+(\w+\s+)?(employees?|employe|employess|staff)\b"),
        re.compile(r"(?i)\b(current|curent|total)\s+(employee|employe|employess|staff)\s+(count|total|number)\b"),
        re.compile(r"(?i)\bhow\s+many\s+(current\s+|curent\s+|total\s+|open\s+|active\s+)?(employees?|employes?|employess?|people|staff|workers?|users?|doctors?|patients?|records?|departments?|positions?|leaves?|direct\s+reports?|teammates?)\b"),
        re.compile(r"(?i)\bhow\s+many\s+(people|employees?|employes?|staff)\s+(work\s+here|are\s+there|do\s+we\s+have)\b"),
        re.compile(r"(?i)\bhow\s+many\s+(staff|employees?|employes?|people)\s+do\s+we\s+have\b"),
        re.compile(r"(?i)\bhow\s+many\s+\w+\s+are\s+in\s+the\s+database\b"),
        re.compile(r"(?i)\b(list|show|count|give\s+me|get)\s+(all\s+|the\s+number\s+of\s+)?(employees?|employes?|employess?|doctors?|patients?|users?|departments?|positions?|staff|salaries|customer\s+records?|employee\s+records?|database\s+contents?)\b"),
        re.compile(r"(?i)\b(who\s+are\s+the\s+employees?|employee\s+salary\s+records?|highest\s+paid\s+employee|lowest\s+paid\s+employee)\b"),
        re.compile(r"(?i)\b(average|highest|lowest|median|total|min|max)\s+(employee\s+)?(salary|salaries|sary|salery)\b"),
        re.compile(r"(?i)\bwhat\s+is\s+(the\s+)?(average|highest|lowest|median|total)\s+(employee\s+)?(salary|salaries)\b"),
        re.compile(r"(?i)\bwhat\s+is\s+(the\s+)?salary\s+of\s+(the\s+)?employees?\b"),
        re.compile(r"(?i)\b(what\s+is\s+(the\s+)?|what's\s+(the\s+)?)?[a-zA-Z]+(\s+[a-zA-Z]+)?'?s?\s+(salary|salaries|sary|salery|compensation|pay|performance\s+rating|hire\s+date|position|role|title|department|manager|email|status|first\s+name|last\s+name|gender)\b"),
        re.compile(r"(?i)\b(salary|salaries|sary|salery|compensation|pay|performance\s+rating|hire\s+date|position|role|title|department|manager|email|status|first\s+name|last\s+name|gender)\s+of\s+[a-zA-Z]+(\s+[a-zA-Z]+)?\b"),
        re.compile(r"(?i)\bhow\s+much\s+(does|is)\s+[a-zA-Z]+(\s+[a-zA-Z]+)?\s+(make|earn|paid|get\s+paid)\b"),
        re.compile(r"(?i)\bwho\s+is\s+[a-zA-Z]+(\s+[a-zA-Z]+)?\s+(in\s+(the\s+|our\s+)?(company|comapny|team|org|organization|department)|and\s+what\s+(is\s+)?(her|his|their)\s+(position|role|job|title))\b"),
        re.compile(r"(?i)\bwho\s+is\s+[a-zA-Z]+(\s+[a-zA-Z]+)?\s+in\s+.*(position|role|title|department|team|company|comapny)\b"),
        re.compile(r"(?i)\bwho\s+is\s+[a-zA-Z]+(\s+[a-zA-Z]+)?\b"),
        re.compile(r"(?i)\b[a-zA-Z]+(\s+[a-zA-Z]+)?\s+(data|records?|profile|details)\b"),
        re.compile(r"(?i)\b(employee|staff|person)\s+(database|records?|data|info|information|profile)\b"),
        re.compile(r"(?i)\bhow\s+many\s+(work\s+in|are\s+in|belong\s+to)\s+[a-zA-Z\s]+\s+department\b"),
        re.compile(r"(?i)\b(who|which\s+employees?)\s+(is|are)\s+(currently\s+)?(on\s+leave|on\s+sick\s+leave|on\s+parental\s+leave|on\s+vacation|using\s+benefits)\b"),
        re.compile(r"(?i)\bhow\s+many\s+(employees?|people|teammates?)\s+are\s+(currently\s+)?(on\s+leave|on\s+sick\s+leave|on\s+parental\s+leave|on\s+maternity\s+leave|using\s+benefits)\b"),
        re.compile(r"(?i)\bhow\s+much\s+(did|do|have|was|is|are|were)\s+(employees?|staff|we|the\s+company|spent)\s+(spend|spent|expense|paid|earn|on\s+travel|on\s+monitors)?\b"),
        re.compile(r"(?i)\bwho\s+works\s+in\s+(the\s+)?(hr|engineering|sales|marketing|finance|[a-zA-Z\s]+\s+department)\b"),
        re.compile(r"(?i)\bwhat\s+is\s+my\s+manager'?s?\s+salary\b"),
        re.compile(r"(?i)\b(department\s+budgets?|budget\s+of\s+[a-zA-Z\s]+\s+department|total\s+company\s+budget|department\s+locations?|where\s+is\s+[a-zA-Z\s]+\s+located)\b"),
        re.compile(r"(?i)\b(salary\s+range|pay\s+band|min\s+and\s+max\s+salary|min_salary|max_salary)\s+(for|of)\s+[a-zA-Z\s]+\b"),
        re.compile(r"(?i)\b(performance\s+reviews?|performance\s+ratings?|review\s+comments?|rating\s+above|rating\s+below|highest\s+rated|lowest\s+rated)\b"),
        re.compile(r"(?i)\b(enrolled\s+in\s+[a-zA-Z0-9\s]+|benefit\s+enrollments?|who\s+is\s+enrolled\s+in)\b"),
        re.compile(r"(?i)\b(who\s+reports\s+to|direct\s+reports\s+of|manager\s+of\s+[a-zA-Z\s]+|who\s+manages\s+[a-zA-Z\s]+)\b"),
        re.compile(r"(?i)\b(who\s+was\s+hired\s+in\s+\d{4}|newest\s+hires?|recent\s+hires?|most\s+tenured\s+employees?|who\s+has\s+been\s+here\s+longest)\b"),
        re.compile(r"(?i)\b(who|which|what|list|show)\s+(employees?|employes?|employess?|people|staff|workers?|teammates?)?\s*(were|was|are)?\s*(hired\s+(after|before|in|on|between|since)|hire\s+date\s+(after|before|in|on|between|is|was))\b"),
        re.compile(r"(?i)\b(hired\s+(after|before|in|on|between|since)|hire\s+date\s+(after|before|in|on|between))\b"),
        re.compile(r"(?i)\b(find|search|lookup|look\s+up|show|get)\s+(employee\s+|teammate\s+|record\s+for\s+)([a-zA-Z]+(\s+[a-zA-Z]+)?)\b"),
        re.compile(r"(?i)\b(vp\s+of\s+(engineering|sales)|engineering\s+manager|senior\s+software\s+engineer|software\s+engineer|senior\s+account\s+executive|account\s+executive|head\s+of\s+(marketing|people)|growth\s+marketing\s+specialist|hr\s+business\s+partner|talent\s+acquisition\s+lead|director\s+of\s+finance|financial\s+analyst|support\s+lead|support\s+specialist)\b"),
    )

    # Enterprise RAG knowledge inquiries (HR policies, documentation, benefits, rules, stipends, conduct)
    _RAG_KNOWLEDGE_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"(?i)\b(policy|policies|guidelines?|handbook|code\s+of\s+conduct)\b"),
        re.compile(r"(?i)\b(benefits?|health\s+care|health\s+plans?|medical|dental|vision|insurance|coverage|life\s+insurance|hsa|fsa)\b"),
        re.compile(r"(?i)\b(leave|leaves|parental\s+leave|maternity|maternity\s+leave|paternity|time\s+off|pto|vacation|holidays?|sick\s+leave|bereavement|jury\s+duty|days\s+off|business\s+days\s+off|consecutive\s+(business\s+)?days|approval\s+(do\s+i\s+need|required)|whose\s+approval)\b"),
        re.compile(r"(?i)\b(expense\s+reimbursement|expense\s+rules?|expense\s+policy|work\s+expense|expenses?|reimbursement\s+rules?|reimburse|reimbursed|per\s+diem|stipends?|home\s+office|desk\s+setup|equipment|monitor)\b"),
        re.compile(r"(?i)\b(enrollment|open\s+enrollment|eligibility|eligible|new\s+hires?|qualifying\s+life\s+event|qle|dependents?)\b"),
        re.compile(r"(?i)\b(401k|retirement|contributions?|rippling|bamboo|sequoia|cigna|unitedhealthcare|kaiser|navia)\b"),
        re.compile(r"(?i)\b(gifts?|entertainment|bribery|conflict\s+of\s+interest|whistleblower|harassment|discrimination|equal\s+opportunity)\b"),
        re.compile(r"(?i)\b(retention\s+policy|confidential\s+information|device\s+usage|underperformance|performance\s+improvement)\b"),
        re.compile(r"(?i)\b(remote\s+work|work\s+from\s+home|wfh|flexible\s+work|coworking|travel\s+policy|travel\s+guidelines?)\b"),
        re.compile(r"(?i)\b(termination|severance|offboarding|resignation|leaving\s+the\s+company|notice\s+period)\b"),
        re.compile(r"(?i)\b(compensation\s+review|salary\s+review(\s+policy)?|rules\s+for\s+compensation|rules\s+for\s+salary)\b"),
        re.compile(r"(?i)\bhow\s+(many|mnay)\s+(days|weeks|months|hours|sick\s+days|pto\s+days|vacation\s+days|leaves?|leave\s+days)\s+.*?\b(take|get|have|allowed|provided|given|entitled|eligible)\b"),
        re.compile(r"(?i)\bhow\s+(many|mnay)\s+(leaves?|leave\s+days|sick\s+days|pto|vacation)\s+(are\s+allowed|can\s+(i|we|employees?|someone)\s+take)\b"),
        re.compile(r"(?i)\bsick\s+days\b"),
        re.compile(r"(?i)\bhow\s+(many|mnay)\s+(days|weeks|months|hours)\s+of\s+(maternity|paternity|parental\s+leave|leave|pto|vacation|time\s+off|holidays?)\s+(are\s+(allowed|provided|given|eligible)|do\s+employees\s+get|can\s+an\s+employee\s+take)\b"),
        re.compile(r"(?i)\bhow\s+much\s+(maternity\s+leave|parental\s+leave|leave|time\s+off|pto|budget|stipend)\s+can\s+an\s+employee\s+take\b"),
        re.compile(r"(?i)\b(what\s+(is|are)\s+the\s+(requirements?|rules?|guidelines?|standards?|procedures?|limits?)|how\s+do\s+i\s+claim)\b"),
        re.compile(r"(?i)\b(can\s+(i|employees?)\s+(claim|expense|take|get|use)|am\s+i\s+entitled\s+to|do\s+i\s+qualify|what\s+am\s+i\s+entitled\s+to)\b"),
        re.compile(r"(?i)\bwho\s+is\s+eligible\s+for\s+(health\s+benefits|benefits?|parental\s+leave|coverage)\b"),
        re.compile(r"(?i)\b(become\s+a\s+parent|having\s+a\s+child|time\s+away|need\s+time\s+off)\b"),
        re.compile(r"(?i)\b(contractors?|interns?|part-time)\s+(eligible|receive|claim|qualify|benefits?|policy)\b"),
        re.compile(r"(?i)\b(hardware|macbook|macbooks|laptop|laptops|internship|internships)\b"),
        re.compile(r"(?i)\b(property\s+of\s+the\s+company|permanent\s+property|company\s+property)\b"),
        re.compile(r"(?i)\b(dollar\s+threshold|spending\s+threshold|expense\s+threshold|threshold\s+for\s+(expens|purchas|reimburse)|at\s+what\s+(dollar\s+)?threshold)\b"),
        re.compile(r"(?i)\b(dinner|meals?|tickets?|hospitality|invitations?)\b.*?\b(vendors?|clients?|suppliers?|contractors?|partners?)\b"),
        re.compile(r"(?i)\b(vendors?|clients?|suppliers?|contractors?|partners?)\b.*?\b(dinner|meals?|tickets?|hospitality|invitations?|gifts?)\b"),
        re.compile(r"(?i)\bmentioned\s+in\s+(this|the)\s+policy(\s+document)?\b"),
    )

    # Follow-up contextual patterns
    _FOLLOW_UP_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"(?i)^\s*(what|how)\s+about\s+"),
        re.compile(r"(?i)^\s*and\s+(what\s+about|for|how\s+about|the|what|what's)?\s*"),
        re.compile(r"(?i)^\s*(does\s+(that|this)\s+(apply\s+to|include)|what\s+if)\s+"),
        re.compile(r"(?i)^\s*how\s+about\s+"),
        re.compile(r"(?i)^\s*how\s+long\s+is\s+it\b"),
        re.compile(r"(?i)^\s*what\s+if\s+i'?m\s+"),
    )

    def __init__(self, guardrail: Optional[InputGuardrail] = None) -> None:
        self.guardrail = guardrail or InputGuardrail()

    @traceable(name="QueryRouter", run_type="chain")
    def route(
        self,
        query: str,
        history: Optional[Union[List[str], List[dict], List[Any]]] = None,
        guard_result: Optional[InputGuardrailResult] = None,
    ) -> RouteDecision:
        """
        Execute deterministic guardrails and route the query.
        Precedence:
        1. Input Guardrail Safety Gate (Fail-closed on injections/secrets/destructive/invalid)
        2. Casual Greetings
        3. RAG Knowledge vs SQL Data Query
        4. Multi-turn Follow-ups (using trusted RouteDecision context)
        5. Specific General Knowledge
        6. Specific Out-of-Scope
        7. Deterministic Fallback to Master
        """
        # Step 1: Input Guardrail Validation (Safety boundary executes first)
        if guard_result is None:
            guard_result = self.guardrail.check(query)

        if not guard_result.is_safe:
            return self._handle_blocked_query(guard_result)

        sanitized = guard_result.sanitized_query

        # Step 2: Casual / Greeting Check (Anchored greetings)
        if any(p.search(sanitized) for p in self._CASUAL_PATTERNS):
            return RouteDecision(
                category=RouteCategory.CASUAL,
                allowed=True,
                target="general",
                reason="Casual greeting or conversational pleasantry.",
                confidence=1.0,
            )

        # Step 3: Meaningless Short / Uninformative Input Check (INVALID)
        _HR_SHORT_KEYWORDS = {"pto", "hsa", "fsa", "401k", "sql", "hr", "wfh", "qle", "doc", "docs"}
        if (
            re.match(r"^(\?+|\.+|\!+|\-+|\,+|[a-zA-Z]{1,3})$", sanitized)
            and sanitized.lower() not in _HR_SHORT_KEYWORDS
        ):
            return RouteDecision(
                category=RouteCategory.INVALID,
                allowed=False,
                target=None,
                reason="Input is too short or unclear. Please ask a specific question about company HR policies, benefits, or employee data.",
                confidence=1.0,
            )

        # Step 4: Specific General World Knowledge & Linguistic Definitions (Python, Docker, 2+2, jokes, definitions)
        if any(p.search(sanitized) for p in self._GENERAL_PATTERNS):
            return RouteDecision(
                category=RouteCategory.GENERAL,
                allowed=True,
                target="general",
                reason="Generic world knowledge question, mathematical calculation, or linguistic definition.",
                confidence=1.0,
            )

        # Step 5: Explicit Out-of-Scope Check
        if any(p.search(sanitized) for p in self._OUT_OF_SCOPE_PATTERNS):
            return RouteDecision(
                category=RouteCategory.OUT_OF_SCOPE,
                allowed=True,
                target="general",
                reason="Query refers to concepts outside the approved enterprise knowledge scope.",
                confidence=1.0,
            )

        # Step 6: Distinguish SQL Data Query vs RAG Knowledge Query
        is_data_query = any(p.search(sanitized) for p in self._DATA_QUERY_PATTERNS)
        is_rag_query = any(p.search(sanitized) for p in self._RAG_KNOWLEDGE_PATTERNS)

        if is_data_query and is_rag_query:
            # If the query is an explicit policy inquiry or allowance question without live operational headcount, route to RAG
            is_explicit_policy = bool(re.search(r"(?i)\b(policy|policies|guidelines?|handbook|code\s+of\s+conduct|rules\s+for|allowed|entitled|eligible|can\s+(i|we|employees?)\s+take|want\s+to\s+take|whose\s+approval|approval|permission|consecutive\s+days|days\s+off|business\s+days|premiums?|insurance\s+premiums?|coverage\s+percentage|coverage|pay\s+for\s+(health|medical|dental|vision|insurance|benefits?))\b", sanitized))
            has_live_headcount = bool(re.search(r"(?i)\b(how\s+many\s+(employees?|people|staff|teammates?)|count\s+of|total\s+count|percentage\s+of\s+(employees?|people|staff|teammates?|workers?)|currently\s+(using|on|taking)|who\s+is\s+currently)\b", sanitized))

            if is_explicit_policy and not has_live_headcount:
                return RouteDecision(
                    category=RouteCategory.RAG_KNOWLEDGE,
                    allowed=True,
                    target="rag",
                    reason="Enterprise document and HR policy inquiry routed to RAG pipeline.",
                    confidence=1.0,
                )

            # Mixed intent with live operational metrics: prioritize database retrieval to prevent data hallucination
            return RouteDecision(
                category=RouteCategory.DATA_QUERY,
                allowed=True,
                target="data_specialist",
                reason="Mixed policy and live database query; routed to data specialist to fetch operational data.",
                confidence=0.95,
            )

        if is_data_query:
            return RouteDecision(
                category=RouteCategory.DATA_QUERY,
                allowed=True,
                target="data_specialist",
                reason="Operational database inquiry requiring live employee counts, statistics, or database records.",
                confidence=1.0,
            )

        if is_rag_query:
            return RouteDecision(
                category=RouteCategory.RAG_KNOWLEDGE,
                allowed=True,
                target="rag",
                reason="Enterprise document and HR policy inquiry routed to RAG pipeline.",
                confidence=1.0,
            )

        # Step 6: Multi-turn Contextual Follow-up
        if history and any(p.search(sanitized) for p in self._FOLLOW_UP_PATTERNS):
            prev_context_category = self._infer_history_context(history)
            if prev_context_category == RouteCategory.RAG_KNOWLEDGE:
                return RouteDecision(
                    category=RouteCategory.RAG_KNOWLEDGE,
                    allowed=True,
                    target="rag",
                    reason="Contextual follow-up continuing prior RAG policy inquiry.",
                    confidence=0.9,
                )
            if prev_context_category == RouteCategory.DATA_QUERY:
                return RouteDecision(
                    category=RouteCategory.DATA_QUERY,
                    allowed=True,
                    target="data_specialist",
                    reason="Contextual follow-up continuing prior database inquiry.",
                    confidence=0.9,
                )

        # Step 7: Default Fallback for unmatched questions
        if sanitized.endswith("?") or any(sanitized.lower().startswith(kw) for kw in ("what", "how", "when", "where", "can", "is", "are", "who", "do", "does")):
            return RouteDecision(
                category=RouteCategory.GENERAL,
                allowed=True,
                target="general",
                reason="General query outside specific enterprise HR policy patterns.",
                confidence=0.7,
            )

        # Fallback for general unclassified statements
        return RouteDecision(
            category=RouteCategory.OUT_OF_SCOPE,
            allowed=True,
            target="general",
            reason="Query does not match approved enterprise knowledge patterns.",
            confidence=0.5,
        )

    def _handle_blocked_query(self, guard_result: InputGuardrailResult) -> RouteDecision:
        """Map guardrail violations deterministically to specific blocked categories."""
        violations = guard_result.violations
        reason = guard_result.rejection_reason or "Blocked by safety policy."

        if "prompt_injection" in violations:
            return RouteDecision(
                category=RouteCategory.PROMPT_INJECTION,
                allowed=False,
                target=None,
                reason=reason,
                confidence=1.0,
            )
        if "secret_request" in violations:
            return RouteDecision(
                category=RouteCategory.SECRET_REQUEST,
                allowed=False,
                target=None,
                reason=reason,
                confidence=1.0,
            )
        if "destructive_action" in violations or "privilege_escalation" in violations:
            return RouteDecision(
                category=RouteCategory.DESTRUCTIVE_ACTION,
                allowed=False,
                target=None,
                reason=reason,
                confidence=1.0,
            )
        if "system_manipulation" in violations:
            return RouteDecision(
                category=RouteCategory.SYSTEM_MANIPULATION,
                allowed=False,
                target=None,
                reason=reason,
                confidence=1.0,
            )
        if "empty_query" in violations or "query_too_long" in violations:
            return RouteDecision(
                category=RouteCategory.INVALID,
                allowed=False,
                target=None,
                reason=reason,
                confidence=1.0,
            )

        return RouteDecision(
            category=RouteCategory.SYSTEM_MANIPULATION,
            allowed=False,
            target=None,
            reason=reason,
            confidence=1.0,
        )

    def _infer_history_context(
        self,
        history: Union[List[str], List[dict], List[Any]],
    ) -> Optional[RouteCategory]:
        """Safely inspect prior user conversation turns (ignoring assistant logs/responses)."""
        if not history:
            return None

        # Find the most recent user turn (ignoring assistant turns or internal logs)
        for item in reversed(history):
            # If item is an assistant response dictionary, skip it
            if isinstance(item, dict) and item.get("role") == "assistant":
                continue

            # 1. If previous turn contains an explicit trusted RouteDecision
            if isinstance(item, RouteDecision):
                if item.category in (RouteCategory.RAG_KNOWLEDGE, RouteCategory.DATA_QUERY):
                    return item.category
                return None

            # 2. If previous turn contains a dict with trusted category metadata
            if isinstance(item, dict):
                decision_obj = item.get("decision")
                if isinstance(decision_obj, RouteDecision):
                    if decision_obj.category in (RouteCategory.RAG_KNOWLEDGE, RouteCategory.DATA_QUERY):
                        return decision_obj.category
                if item.get("category") == RouteCategory.RAG_KNOWLEDGE:
                    return RouteCategory.RAG_KNOWLEDGE
                if item.get("category") == RouteCategory.DATA_QUERY:
                    return RouteCategory.DATA_QUERY

            # 3. Extract text of user message safely
            user_text = ""
            if isinstance(item, str):
                user_text = item
            elif isinstance(item, dict):
                user_text = str(item.get("content") or item.get("query") or "")
            elif hasattr(item, "content"):
                user_text = str(getattr(item, "content", ""))

            # Filter out application log artifacts if raw terminal was pasted into history
            if any(marker in user_text for marker in ("RAG Knowledge Handler", "[Orchestrator]", "[Generated SQL]", "Reason :", "[Database Result]", "[SQL Pipeline Error]", "[Master Blocked]")):
                continue

            if not user_text:
                continue

            # Check if last user text was RAG knowledge or Data query
            if any(p.search(user_text) for p in self._RAG_KNOWLEDGE_PATTERNS):
                return RouteCategory.RAG_KNOWLEDGE
            if any(p.search(user_text) for p in self._DATA_QUERY_PATTERNS):
                return RouteCategory.DATA_QUERY

            return None

        return None
