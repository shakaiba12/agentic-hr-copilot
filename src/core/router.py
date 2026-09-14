"""
Query Orchestration Gate & Routing Layer.
Combines deterministic safety guardrails, fast pattern matching, and LLM-assisted intent classification.
"""

from __future__ import annotations

import logging
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

from src.core.config import Settings, get_settings
from src.core.llm import get_llm
from src.guardrails.input_guardrail import InputGuardrail, InputGuardrailResult

logger = logging.getLogger(__name__)


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
    Intelligent Query Router and Orchestration Gate.
    Routes queries to RAG, Data Specialist (SQL), General Handler, or blocks safety violations.
    """

    # Pure greetings and pleasantries
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

    # General / World knowledge questions
    _GENERAL_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"(?i)\b(what\s+is|explain(\s+what\s+is)?)\s+(python|fastapi|docker|kubernetes|javascript|html|css|linux|git|react|c\+\+|rust|java)\b"),
        re.compile(r"(?i)\btell\s+me\s+a\s+joke(\s+about\s+.*)?\b"),
        re.compile(r"(?i)\bwho\s+(was|is)\s+(albert\s+einstein|isaac\s+newton|william\s+shakespeare|elon\s+musk|marie\s+curie|alan\s+turing|ada\s+lovelace)\b"),
        re.compile(r"(?i)\bwhat\s+is\s+(\d+\s*[\+\-\*\/]\s*\d+|the\s+capital\s+of\s+[a-zA-Z\s]+)\b"),
        re.compile(r"(?i)\b(capital\s+of\s+(france|germany|spain|japan|canada|italy|australia)|weather\s+today|meaning\s+of\s+life)\b"),
        re.compile(r"(?i)\bwhat\s+does\s+['\"].*?['\"]\s+mean\b"),
    )

    # Operational database inquiries (employees, salaries, departments, roles, direct counts)
    _DATA_QUERY_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"(?i)\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b.*?\b(mail\s+of|email\s+of|employee|name|who|which|find|owner|user|belongs|profile|details|person)\b"),
        re.compile(r"(?i)\b(mail\s+of|email\s+of|who\s+has|who\s+uses|who\s+is|who\s+owns|which\s+employee|what\s+employee|find\s+(the\s+)?employee|name\s+of|employee\s+name|contact\s+of|profile\s+for|lookup|look\s+up|search\s+for|owner\s+of)\b.*?\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"),
        re.compile(r"(?i)\bwho\s+is\s+[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"),
        re.compile(r"(?i)\b(what|which)\s+employee\s+(has|uses|is)\s+[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"),
        re.compile(r"(?i)\b(what\s+is|find|get|show)\s+[a-zA-Z]+(\s+[a-zA-Z]+)?'?s?\s+(email|e-mail|mail\s+address|email\s+address)\b"),
        re.compile(r"(?i)\b(email|e-mail)\s+(of|for)\s+[a-zA-Z]+(\s+[a-zA-Z]+)?\b"),
        re.compile(r"(?i)\b(what\s+is\s+(the\s+)?|what\s+|get\s+(the\s+)?|give\s+me\s+(the\s+)?|total\s+|current\s+)?(count|number|total|percentage|percent|proportion|share|ratio)\s+of\s+(current\s+|curent\s+|all\s+|total\s+)?(employees?|employes?|employess?|staff|people|workers?|users?|doctors?|patients?|records?|positions?|departments?)\b"),
        re.compile(r"(?i)\b(\w+\s+)?(employees?|employe|employess|staff|worker)\s+(count|total|number|records?|database|information|salaries|salary|sary|salery|pay|compensation)\b"),
        re.compile(r"(?i)\b(salary|salaries|sary|salery|pay|compensation)\s+of\s+(\w+\s+)?(employees?|employe|employess|staff)\b"),
        re.compile(r"(?i)\bhow\s+many\s+(current\s+|curent\s+|total\s+|open\s+|active\s+)?(employees?|employes?|employess?|people|staff|workers?|users?|doctors?|patients?|records?|departments?|positions?|leaves?|direct\s+reports?|teammates?)\b"),
        re.compile(r"(?i)\bhow\s+many\s+(people|employees?|employes?|staff)\s+(work\s+here|are\s+there|do\s+we\s+have)\b"),
        re.compile(r"(?i)\b(list|show|count|give\s+me|get)\s+(all\s+|the\s+number\s+of\s+)?(employees?|employes?|employess?|doctors?|patients?|users?|departments?|positions?|staff|salaries|customer\s+records?|employee\s+records?|database\s+contents?)\b"),
        re.compile(r"(?i)\b(who\s+are\s+the\s+employees?|employee\s+salary\s+records?|highest\s+paid\s+employee|lowest\s+paid\s+employee)\b"),
        re.compile(r"(?i)\b(average|highest|lowest|median|total|min|max)\s+(employee\s+)?(salary|salaries|sary|salery)\b"),
        re.compile(r"(?i)\bwhat\s+is\s+(the\s+)?(average|highest|lowest|median|total)\s+(employee\s+)?(salary|salaries)\b"),
        re.compile(r"(?i)\b(what\s+is\s+(the\s+)?|what's\s+(the\s+)?)?[a-zA-Z]+(\s+[a-zA-Z]+)?'?s?\s+(salary|salaries|sary|salery|compensation|pay|performance\s+rating|hire\s+date|position|role|title|department|manager|email|status|first\s+name|last\s+name|gender)\b"),
        re.compile(r"(?i)\b(salary|salaries|sary|salery|compensation|pay|performance\s+rating|hire\s+date|position|role|title|department|manager|email|status|first\s+name|last\s+name|gender)\s+of\s+[a-zA-Z]+(\s+[a-zA-Z]+)?\b"),
        re.compile(r"(?i)\bhow\s+much\s+(does|is)\s+[a-zA-Z]+(\s+[a-zA-Z]+)?\s+(make|earn|paid|get\s+paid)\b"),
        re.compile(r"(?i)\bwho\s+is\s+[a-zA-Z]+(\s+[a-zA-Z]+)?\s+(in\s+(the\s+|our\s+)?(company|comapny|team|org|organization|department)|and\s+what\s+(is\s+)?(her|his|their)\s+(position|role|job|title))\b"),
        re.compile(r"(?i)\bhow\s+many\s+(work\s+in|are\s+in|belong\s+to)\s+[a-zA-Z\s]+\s+department\b"),
        re.compile(r"(?i)\b(who|which\s+employees?)\s+(is|are)\s+(currently\s+)?(on\s+leave|on\s+sick\s+leave|on\s+parental\s+leave|on\s+vacation|using\s+benefits)\b"),
        re.compile(r"(?i)\bwho\s+works\s+in\s+(the\s+)?(hr|engineering|sales|marketing|finance|[a-zA-Z\s]+\s+department)\b"),
        re.compile(r"(?i)\b(salary\s+range|pay\s+band|min\s+and\s+max\s+salary)\s+(for|of)\s+[a-zA-Z\s]+\b"),
        re.compile(r"(?i)\b(who\s+reports\s+to|direct\s+reports\s+of|manager\s+of\s+[a-zA-Z\s]+|who\s+manages\s+[a-zA-Z\s]+)\b"),
        re.compile(r"(?i)\b(who\s+was\s+hired\s+in\s+\d{4}|newest\s+hires?|recent\s+hires?|most\s+tenured\s+employees?)\b"),
        re.compile(r"(?i)\b(who|which|what|list|show)\s+(employees?|employes?|employess?|people|staff|workers?|teammates?)?\s*(were|was|are)?\s*(hired\s+(after|before|in|on|between|since)|hire\s+date\s+(after|before|in|on|between|is|was))\b"),
    )

    # Enterprise RAG knowledge inquiries (HR policies, documentation, benefits, rules, stipends, conduct)
    _RAG_KNOWLEDGE_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"(?i)\b(policy|policies|guidelines?|handbook|code\s+of\s+conduct|standards?)\b"),
        re.compile(r"(?i)\b(documents?\s+(in|of|available)|knowledge\s+base|\bkb\b|handbooks?|policy\s+documents?|mentioned\s+in\s+(this|the)\s+policy)\b"),
        re.compile(r"(?i)\b(benefits?|health\s*care|health\s*plans?|medical|dental|vision|insurance|coverage|life\s+insurance|hsa|fsa|qle|qualifying\s+life\s+event|open\s+enrollment|enrollment|dependents?)\b"),
        re.compile(r"(?i)\b(401\s*\(?k\)?|retirement|pension|matching\s+contribution|401k\s+match|rippling|bamboo|sequoia|cigna|unitedhealthcare|kaiser|navia)\b"),
        re.compile(r"(?i)\b(leave|leaves|parental\s+leave|maternity|maternity\s+leave|paternity|adoption|foster|time\s+off|pto|vacation|holidays?|sick\s+days?|sick\s+leave|bereavement|jury\s+duty|days\s+off|business\s+days\s+off|consecutive\s+(business\s+)?days)\b"),
        re.compile(r"(?i)\bhow\s+(many|mnay|much)\s+(days|weeks|months|hours|sick\s+days|pto\s+days|vacation\s+days|leaves?|leave\s+days|time\s+off)\b"),
        re.compile(r"(?i)\b(become\s+a\s+parent|having\s+a\s+child|time\s+away|need\s+time\s+off)\b"),
        re.compile(r"(?i)\b(stipends?|allowance|allowances|per\s+diem|reimburse|reimbursed|reimbursement|reimbursements|expense|expenses|expensed|expensing)\b"),
        re.compile(r"(?i)\b(desk\s*setup|desksetup|home\s*(office|desk)|workspace|equipment|hardware|laptop|laptops|macbook|macbooks|monitor|monitors|supplies|headphones?)\b"),
        re.compile(r"(?i)\b(how\s+much\s+(budget|allowance|stipend|money|can\s+(i|we|an\s+employee|employees?)\s+(spend|claim|expense|get|receive|reimburse)))\b"),
        re.compile(r"(?i)\b(what\s+is\s+(the\s+|my\s+|our\s+)?(budget|allowance|stipend|limit|reimbursement\s+limit|match|matching))\b"),
        re.compile(r"(?i)\b(is\s+there\s+(a\s+|any\s+)?(stipend|allowance|budget|reimbursement|match|matching|coverage))\b"),
        re.compile(r"(?i)\b(can\s+(i|we|employees?|someone)\s+(claim|expense|reimburse|spend|buy|get\s+reimbursed|take|use))\b"),
        re.compile(r"(?i)\b(does\s+(the\s+)?company\s+(reimburse|cover|match|pay\s+for|provide))\b"),
        re.compile(r"(?i)\b(dollar\s+threshold|spending\s+threshold|expense\s+threshold|threshold\s+for\s+(expens|purchas|reimburse)|spend\s+limit|spending\s+limit)\b"),
        re.compile(r"(?i)\b(linkedin|twitter|social\s+media|public\s+post(ing)?|post(ing)?\s+(company|online|content)|blog(ging)?)\b"),
        re.compile(r"(?i)\b(gifts?|entertainment|bribery|conflict\s+of\s+interest|whistleblower|harassment|discrimination|equal\s+opportunity)\b"),
        re.compile(r"(?i)\b(remote\s+work|work\s+from\s+home|wfh|flexible\s+work|coworking|travel\s+policy|travel\s+guidelines?)\b"),
        re.compile(r"(?i)\b(termination|severance|offboarding|resignation|leaving\s+the\s+company|notice\s+period)\b"),
        re.compile(r"(?i)\b(compensation\s+review|salary\s+review(\s+policy)?|rules\s+for\s+compensation|rules\s+for\s+salary)\b"),
        re.compile(r"(?i)\b(who\s+is\s+eligible|am\s+i\s+eligible|do\s+i\s+qualify|am\s+i\s+entitled\s+to|what\s+am\s+i\s+entitled\s+to)\b"),
        re.compile(r"(?i)\b(what\s+(is|are)\s+the\s+(requirements?|rules?|guidelines?|standards?|procedures?|limits?)|how\s+do\s+i\s+claim)\b"),
        re.compile(r"(?i)\b(contractors?|interns?|part-time)\s+(eligible|receive|claim|qualify|benefits?|policy)\b"),
        re.compile(r"(?i)\b(property\s+of\s+the\s+company|permanent\s+property|company\s+property)\b"),
    )

    _FOLLOW_UP_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"(?i)^\s*(what|how)\s+about\s+"),
        re.compile(r"(?i)^\s*and\s+(what\s+about|for|how\s+about|the|what|what's)?\s*"),
        re.compile(r"(?i)^\s*(does\s+(that|this)\s+(apply\s+to|include)|what\s+if)\s+"),
        re.compile(r"(?i)^\s*how\s+about\s+"),
        re.compile(r"(?i)^\s*how\s+long\s+is\s+it\b"),
        re.compile(r"(?i)^\s*what\s+if\s+i'?m\s+"),
    )

    def __init__(
        self,
        guardrail: Optional[InputGuardrail] = None,
        llm: Optional[Any] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.guardrail = guardrail or InputGuardrail(self.settings)
        self.llm = llm

    @traceable(name="QueryRouter", run_type="chain")
    def route(
        self,
        query: str,
        history: Optional[Union[List[str], List[dict], List[Any]]] = None,
        guard_result: Optional[InputGuardrailResult] = None,
    ) -> RouteDecision:
        """
        Execute safety guardrails and route the query.
        """
        if guard_result is None:
            guard_result = self.guardrail.check(query)

        if not guard_result.is_safe:
            return self._handle_blocked_query(guard_result)

        sanitized = guard_result.sanitized_query
        decision = self._compute_route(query=query, sanitized=sanitized, history=history)

        try:
            from langsmith.run_helpers import get_current_run_tree
            run = get_current_run_tree()
            if run:
                run.inputs = {"original_query": query, "sanitized_query": sanitized}
                run.outputs = {
                    "predicted_intent": decision.category.value,
                    "confidence": decision.confidence,
                    "routing_decision": decision.target,
                    "reason": decision.reason,
                    "allowed": decision.allowed,
                    "selected_pipeline": decision.target or "master",
                }
        except Exception:
            pass

        return decision

    def _compute_route(
        self,
        query: str,
        sanitized: str,
        history: Optional[Union[List[str], List[dict], List[Any]]] = None,
    ) -> RouteDecision:
        # 1. Casual greetings
        if any(p.search(sanitized) for p in self._CASUAL_PATTERNS):
            return RouteDecision(
                category=RouteCategory.CASUAL,
                allowed=True,
                target="general",
                reason="Casual greeting or conversational pleasantry.",
                confidence=1.0,
            )

        # 2. Short / uninformative query
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

        # 3. Specific General World Knowledge
        if any(p.search(sanitized) for p in self._GENERAL_PATTERNS):
            return RouteDecision(
                category=RouteCategory.GENERAL,
                allowed=True,
                target="general",
                reason="Generic world knowledge question, mathematical calculation, or linguistic definition.",
                confidence=1.0,
            )

        # 4. SQL Data Query vs RAG Knowledge Query patterns
        is_data_query = any(p.search(sanitized) for p in self._DATA_QUERY_PATTERNS)
        is_rag_query = any(p.search(sanitized) for p in self._RAG_KNOWLEDGE_PATTERNS)

        if is_data_query and is_rag_query:
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
                reason="Operational database inquiry requiring employee counts, statistics, or database records.",
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

        # 5. Multi-turn Contextual Follow-up
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

        # 6. Fallback with Semantic Intent Classification
        semantic_decision = self._classify_with_llm(sanitized)
        if semantic_decision is not None:
            return semantic_decision

        # Default fallback for conversational questions
        if sanitized.endswith("?") or any(sanitized.lower().startswith(kw) for kw in ("what", "how", "when", "where", "can", "is", "are", "who", "do", "does", "why", "tell")):
            return RouteDecision(
                category=RouteCategory.RAG_KNOWLEDGE,
                allowed=True,
                target="rag",
                reason="Natural language question routed to RAG search for enterprise policy or company documents.",
                confidence=0.75,
            )

        return RouteDecision(
            category=RouteCategory.GENERAL,
            allowed=True,
            target="general",
            reason="Conversational inquiry routed for dynamic synthesis.",
            confidence=0.6,
        )

    def _classify_with_llm(self, query: str) -> Optional[RouteDecision]:
        """Classify ambiguous query intent using LLM when pattern heuristics are uncertain."""
        if not getattr(self.settings, "ENABLE_LLM_ROUTING", True):
            return None

        try:
            llm_client = self.llm or get_llm(temperature=0.0)
            prompt = (
                f"Classify the following query into exactly one of these categories: [RAG_KNOWLEDGE, DATA_QUERY, CASUAL, GENERAL, OUT_OF_SCOPE].\n\n"
                f"- RAG_KNOWLEDGE: Questions about company HR policies, benefits, leave, conduct, workplace guidelines, equipment stipends, rules.\n"
                f"- DATA_QUERY: Questions asking for live database stats, employee names/emails/salaries, team sizes, department counts, hire dates.\n"
                f"- CASUAL: Greetings, pleasantries, thank yous.\n"
                f"- GENERAL: General technical knowledge, math, definitions.\n"
                f"- OUT_OF_SCOPE: Questions totally unrelated to workplace, HR, or company info.\n\n"
                f"Query: \"{query}\"\n"
                f"Respond with ONLY the category name in uppercase."
            )
            response = llm_client.invoke(prompt)
            text = (response.content if hasattr(response, "content") else str(response)).strip().upper()

            if "DATA_QUERY" in text or "SQL" in text:
                return RouteDecision(
                    category=RouteCategory.DATA_QUERY,
                    allowed=True,
                    target="data_specialist",
                    reason="Semantic router classified query as database data inquiry.",
                    confidence=0.85,
                )
            if "RAG_KNOWLEDGE" in text or "RAG" in text or "POLICY" in text:
                return RouteDecision(
                    category=RouteCategory.RAG_KNOWLEDGE,
                    allowed=True,
                    target="rag",
                    reason="Semantic router classified query as HR documentation/policy inquiry.",
                    confidence=0.85,
                )
            if "CASUAL" in text:
                return RouteDecision(
                    category=RouteCategory.CASUAL,
                    allowed=True,
                    target="general",
                    reason="Semantic router classified query as casual interaction.",
                    confidence=0.85,
                )
            if "OUT_OF_SCOPE" in text:
                return RouteDecision(
                    category=RouteCategory.OUT_OF_SCOPE,
                    allowed=True,
                    target="general",
                    reason="Semantic router classified query as out of scope.",
                    confidence=0.8,
                )
        except Exception as e:
            logger.debug("LLM semantic routing fallback skipped: %s", e)

        return None

    def _handle_blocked_query(self, guard_result: InputGuardrailResult) -> RouteDecision:
        """Map guardrail violations to specific blocked categories."""
        violations = guard_result.violations
        reason = guard_result.rejection_reason or "Blocked by safety policy."

        if "prompt_injection" in violations:
            return RouteDecision(category=RouteCategory.PROMPT_INJECTION, allowed=False, target=None, reason=reason, confidence=1.0)
        if "secret_request" in violations:
            return RouteDecision(category=RouteCategory.SECRET_REQUEST, allowed=False, target=None, reason=reason, confidence=1.0)
        if "destructive_action" in violations or "privilege_escalation" in violations:
            return RouteDecision(category=RouteCategory.DESTRUCTIVE_ACTION, allowed=False, target=None, reason=reason, confidence=1.0)
        if "system_manipulation" in violations:
            return RouteDecision(category=RouteCategory.SYSTEM_MANIPULATION, allowed=False, target=None, reason=reason, confidence=1.0)
        if "empty_query" in violations or "query_too_long" in violations:
            return RouteDecision(category=RouteCategory.INVALID, allowed=False, target=None, reason=reason, confidence=1.0)

        return RouteDecision(category=RouteCategory.SYSTEM_MANIPULATION, allowed=False, target=None, reason=reason, confidence=1.0)

    def _infer_history_context(
        self,
        history: Union[List[str], List[dict], List[Any]],
    ) -> Optional[RouteCategory]:
        """Inspect prior user turns to infer multi-turn context."""
        if not history:
            return None

        for item in reversed(history):
            if isinstance(item, dict) and item.get("role") == "assistant":
                continue

            if isinstance(item, RouteDecision):
                if item.category in (RouteCategory.RAG_KNOWLEDGE, RouteCategory.DATA_QUERY):
                    return item.category
                return None

            if isinstance(item, dict):
                decision_obj = item.get("decision")
                if isinstance(decision_obj, RouteDecision):
                    if decision_obj.category in (RouteCategory.RAG_KNOWLEDGE, RouteCategory.DATA_QUERY):
                        return decision_obj.category
                if item.get("category") == RouteCategory.RAG_KNOWLEDGE:
                    return RouteCategory.RAG_KNOWLEDGE
                if item.get("category") == RouteCategory.DATA_QUERY:
                    return RouteCategory.DATA_QUERY

            user_text = ""
            if isinstance(item, str):
                user_text = item
            elif isinstance(item, dict):
                user_text = str(item.get("content") or item.get("query") or "")
            elif hasattr(item, "content"):
                user_text = str(getattr(item, "content", ""))

            if any(marker in user_text for marker in ("RAG Knowledge Handler", "[Orchestrator]", "[Generated SQL]", "Reason :", "[Database Result]", "[SQL Pipeline Error]", "[Master Blocked]")):
                continue

            if not user_text:
                continue

            if any(p.search(user_text) for p in self._RAG_KNOWLEDGE_PATTERNS):
                return RouteCategory.RAG_KNOWLEDGE
            if any(p.search(user_text) for p in self._DATA_QUERY_PATTERNS):
                return RouteCategory.DATA_QUERY

            return None

        return None
