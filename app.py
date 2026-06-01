#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MONOLITHE COGNITIF 

Invariants majeurs :
- Toute mémoire a : id unique, vector normalisé, kind ∈ {"episodic","semantic","meta","system"}.
- Toute relation de graphe a : source/target existants, weight ∈ ]0,1], normalisation par source.
- Toute requête LLM passe par LLMEngine (pas d'appel brut à requests/httpx ailleurs).
- Toute mise à jour de mémoire passe par MemoryEngine (pas de mutation sauvage).
- Toute décision passe par PolicyEngine (priorisation explicite).
"""

import json
import math
import sqlite3
import time
import uuid
import logging
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import requests
from pydantic import BaseModel, ValidationError

# ============================================================
# CONFIG & LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("monolithe_10_10")

@dataclass(frozen=True)
class Config:
    # LLM / Embeddings
    ollama_url: str = "http://localhost:11434"
    model: str = "qwen2.5:latest"
    embed_model: str = "nomic-embed-text"

    # Persistence
    db_path: str = "monolithe_10_10.db"

    # Memory
    short_term_size: int = 32
    max_memories: int = 150_000
    decay_rate: float = 0.990
    touch_boost: float = 0.12
    min_energy: float = 0.01
    max_energy: float = 5.0

    # Retrieval weights
    top_k: int = 12
    sim_weight: float = 0.53
    energy_weight: float = 0.22
    recency_weight: float = 0.13
    freq_weight: float = 0.08
    emotion_weight: float = 0.04

    # Graph
    spreading_steps: int = 5
    spreading_decay: float = 0.58

    # Meta-cognition
    abstraction_interval: int = 6
    introspection_interval: int = 5
    abstraction_temp: float = 0.09

    # Evaluation
    coherence_threshold: float = 0.35

CFG = Config()

# ============================================================
# SCHEMAS
# ============================================================

class SynthesisSchema(BaseModel):
    meta_concept: str
    inevitabilite_rationnelle: str
    phenomene_deduit: str
    bulles_cible_ids: List[str]
    coefficient_precision: float

class SelfEvalSchema(BaseModel):
    coherence: float
    humility: float
    risk_of_error: float
    summary: str

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class MemoryRecord:
    id: int
    text: str
    vector: np.ndarray
    kind: str
    energy: float
    created_step: int
    last_access_step: int
    access_count: int
    emotional_weight: float = 0.0

@dataclass
class Goal:
    id: str
    description: str
    priority: float
    created_at: float
    completed: bool = False

@dataclass
class ConceptNode:
    id: str
    label: str
    content: str
    tags: List[str]
    created_at: float = field(default_factory=time.time)

@dataclass
class Relation:
    id: str
    source: str
    target: str
    relation_type: str
    weight: float

# ============================================================
# EMBEDDING ENGINE
# ============================================================

class EmbeddingEngine:
    """
    Invariant : embed(text) retourne toujours un vecteur float32 normalisé (norme > 0 ou vecteur nul).
    """

    def __init__(self):
        self.url = CFG.ollama_url

    def _remote_embed(self, text: str) -> Optional[np.ndarray]:
        try:
            r = requests.post(
                f"{self.url}/api/embeddings",
                json={"model": CFG.embed_model, "prompt": text},
                timeout=15
            )
            r.raise_for_status()
            emb = np.array(r.json()["embedding"], dtype=np.float32)
            n = np.linalg.norm(emb)
            return emb / n if n > 0 else emb
        except Exception as e:
            log.warning(f"Remote embedding failed → fallback ({e})")
            return None

    def _fallback_embed(self, text: str, dim: int = 768) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float32)
        for tok in text.lower().split()[:60]:
            h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h & 1) else -1.0
            vec[idx] += sign * 0.8
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, text: str) -> np.ndarray:
        emb = self._remote_embed(text)
        return emb if emb is not None else self._fallback_embed(text)

# ============================================================
# MEMORY ENGINE
# ============================================================

class MemoryEngine:
    """
    Invariants :
    - self.memories[id].vector est toujours float32 et (quasi) normalisé.
    - kind ∈ {"episodic","semantic","meta","system"}.
    - energy ∈ [CFG.min_energy, CFG.max_energy].
    """

    VALID_KINDS = {"episodic", "semantic", "meta", "system"}

    def __init__(self):
        self.conn = sqlite3.connect(CFG.db_path, isolation_level=None)
        self.embedder = EmbeddingEngine()
        self.memories: Dict[int, MemoryRecord] = {}
        self.short_term: deque[int] = deque(maxlen=CFG.short_term_size)
        self.current_step: int = 0
        self.next_id: int = 0
        self._init_db()
        self._load()

    def _init_db(self):
        self.conn.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                text TEXT,
                vector TEXT,
                kind TEXT,
                energy REAL,
                created_step INTEGER,
                last_access_step INTEGER,
                access_count INTEGER,
                emotional_weight REAL
            );
        """)

    def _load(self):
        rows = self.conn.execute("SELECT * FROM memories").fetchall()
        for row in rows:
            vec = np.array(json.loads(row[2]), dtype=np.float32)
            m = MemoryRecord(
                id=row[0], text=row[1], vector=vec, kind=row[3],
                energy=row[4], created_step=row[5], last_access_step=row[6],
                access_count=row[7], emotional_weight=row[8]
            )
            self.memories[m.id] = m
            self.next_id = max(self.next_id, m.id + 1)
        log.info(f"Loaded {len(self.memories)} memories from DB")

    def tick(self):
        self.current_step += 1
        to_delete = []
        for m in list(self.memories.values()):
            delta = max(0, self.current_step - m.last_access_step)
            m.energy *= (CFG.decay_rate ** delta)
            m.energy = max(CFG.min_energy, min(CFG.max_energy, m.energy))
            if m.energy <= CFG.min_energy * 1.4:
                to_delete.append(m.id)

        for mid in to_delete:
            del self.memories[mid]
            self.conn.execute("DELETE FROM memories WHERE id = ?", (mid,))

    def _validate_kind(self, kind: str):
        if kind not in self.VALID_KINDS:
            raise ValueError(f"Invalid memory kind: {kind}")

    def add(self, text: str, kind: str = "semantic", emotional_weight: float = 0.0):
        self._validate_kind(kind)

        if len(self.memories) >= CFG.max_memories:
            lowest = min(self.memories.values(), key=lambda m: m.energy)
            del self.memories[lowest.id]
            self.conn.execute("DELETE FROM memories WHERE id = ?", (lowest.id,))

        vec = self.embedder.embed(text)
        m = MemoryRecord(
            id=self.next_id, text=text, vector=vec, kind=kind,
            energy=1.0, created_step=self.current_step,
            last_access_step=self.current_step, access_count=1,
            emotional_weight=emotional_weight
        )
        self.memories[m.id] = m
        self.short_term.append(m.id)

        self.conn.execute(
            "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?)",
            (m.id, m.text, json.dumps(m.vector.tolist()), m.kind,
             m.energy, m.created_step, m.last_access_step,
             m.access_count, m.emotional_weight)
        )
        self.next_id += 1

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def retrieve(self, query: str, k: int = CFG.top_k) -> List[MemoryRecord]:
        if not self.memories:
            return []
        qv = self.embedder.embed(query)
        scored = []
        for m in self.memories.values():
            sim = self._cosine(qv, m.vector)
            delta = max(0, self.current_step - m.last_access_step)
            recency = 1.0 / (1.0 + float(delta))
            freq = math.log1p(m.access_count)
            score = (CFG.sim_weight * sim +
                     CFG.energy_weight * m.energy +
                     CFG.recency_weight * recency +
                     CFG.freq_weight * freq +
                     CFG.emotion_weight * m.emotional_weight)
            scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [m for _, m in scored[:k]]

        for m in top:
            m.energy = min(CFG.max_energy, m.energy + CFG.touch_boost)
            m.last_access_step = self.current_step
            m.access_count += 1
            self.conn.execute(
                "UPDATE memories SET energy=?, last_access_step=?, access_count=? WHERE id=?",
                (m.energy, m.last_access_step, m.access_count, m.id)
            )
        return top

# ============================================================
# SEMANTIC GRAPH
# ============================================================

class SemanticGraph:
    """
    Invariants :
    - Tout node.id est unique.
    - Pour chaque source, la somme des weights sortants est ≈ 1.0 (normalisation).
    """

    def __init__(self):
        self.nodes: Dict[str, ConceptNode] = {}
        self.relations: Dict[str, Relation] = {}
        self.outgoing: Dict[str, List[str]] = defaultdict(list)
        self.weights: Dict[Tuple[str, str], float] = {}

    def create_node(self, label: str, content: str, tags: Optional[List[str]] = None) -> str:
        nid = uuid.uuid4().hex[:12]
        node = ConceptNode(nid, label, content, tags or [])
        self.nodes[nid] = node
        return nid

    def connect(self, source: str, target: str, relation_type: str = "related", weight: float = 1.0):
        if source not in self.nodes or target not in self.nodes:
            return None
        weight = max(1e-6, min(1.0, weight))
        rid = uuid.uuid4().hex[:10]
        rel = Relation(rid, source, target, relation_type, weight)
        self.relations[rid] = rel
        self.outgoing[source].append(target)
        self.weights[(source, target)] = weight
        self._normalize(source)
        return rid

    def _normalize(self, source: str):
        targets = self.outgoing.get(source, [])
        if not targets:
            return
        total = sum(self.weights.get((source, t), 0.0) for t in targets)
        if total > 0:
            for t in targets:
                self.weights[(source, t)] /= total

    def spreading_activation(self, start_nodes: List[str], steps: int = CFG.spreading_steps) -> Dict[str, float]:
        scores: Dict[str, float] = defaultdict(float)
        for nid in start_nodes:
            if nid in self.nodes:
                scores[nid] += 1.0

        for _ in range(steps):
            next_scores = defaultdict(float)
            for nid, value in scores.items():
                next_scores[nid] += value * (1 - CFG.spreading_decay)
                for target in self.outgoing.get(nid, []):
                    w = self.weights.get((nid, target), 0.0)
                    next_scores[target] += value * CFG.spreading_decay * w
            scores = next_scores
        return dict(scores)

# ============================================================
# CONTRADICTION ENGINE
# ============================================================

class ContradictionEngine:
    NEGATIONS = [
        ("always", "never"), ("true", "false"), ("possible", "impossible"),
        ("tout", "rien"), ("jamais", "toujours"), ("vivant", "mort")
    ]

    def detect(self, memories: List[MemoryRecord]) -> List[Tuple[str, str]]:
        texts = [m.text.lower() for m in memories]
        contradictions = []
        for i, a in enumerate(texts):
            for b in texts[i+1:]:
                for x, y in self.NEGATIONS:
                    if (x in a and y in b) or (y in a and x in b):
                        contradictions.append((a, b))
        return contradictions

# ============================================================
# EXECUTIVE & POLICY ENGINE
# ============================================================

class ExecutiveSystem:
    def __init__(self):
        self.goals: Dict[str, Goal] = {}

    def add_goal(self, description: str, priority: float = 1.0):
        gid = uuid.uuid4().hex[:10]
        self.goals[gid] = Goal(gid, description, priority, time.time())

    def active_goals(self) -> List[Goal]:
        active = [g for g in self.goals.values() if not g.completed]
        active.sort(key=lambda g: g.priority, reverse=True)
        return active

class PolicyEngine:
    """
    Décide comment pondérer :
    - profondeur vs concision
    - prudence vs assertivité
    - exploration vs exploitation
    """

    def decide_temperature(self, self_eval: Optional[SelfEvalSchema]) -> float:
        if self_eval is None:
            return 0.7
        if self_eval.risk_of_error > 0.6:
            return 0.35
        if self_eval.coherence > 0.8:
            return 0.65
        return 0.5

    def decide_style_tag(self, self_eval: Optional[SelfEvalSchema]) -> str:
        if self_eval is None:
            return "neutre"
        if self_eval.humility > 0.7:
            return "prudent"
        if self_eval.coherence > 0.8:
            return "affirmé"
        return "équilibré"

# ============================================================
# LLM ENGINE
# ============================================================

class LLMEngine:
    """
    Invariant : toutes les interactions LLM passent par cette classe.
    """

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        try:
            r = requests.post(
                f"{CFG.ollama_url}/api/chat",
                json={
                    "model": CFG.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                    "options": {"temperature": temperature}
                },
                timeout=(10, 300)
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
        except requests.exceptions.Timeout:
            log.error("LLM timeout → fallback response")
            return "LLM temporairement indisponible (timeout)."
        except Exception as e:
            log.error(f"LLM failed: {e}")
            return "Je rencontre une instabilité cognitive temporaire."

    def generate_json(self, system_prompt: str, user_payload: Any, schema: BaseModel, temperature: float = 0.2) -> Optional[BaseModel]:
        try:
            r = requests.post(
                f"{CFG.ollama_url}/api/chat",
                json={
                    "model": CFG.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                    ],
                    "stream": False,
                    "options": {"temperature": temperature}
                },
                timeout=120
                
            )
            r.raise_for_status()
            raw = r.json()["message"]["content"]
            data = json.loads(raw)
            return schema.model_validate(data)
        
        except Exception as e:
            log.warning(f"JSON LLM failed: {e}")
            return None

# ============================================================
# COGNITIVE MONOLITH
# ============================================================

class CognitiveMonolith:
    def __init__(self):
        self.memory = MemoryEngine()
        self.graph = SemanticGraph()
        self.contradictions = ContradictionEngine()
        self.executive = ExecutiveSystem()
        self.policy = PolicyEngine()
        self.llm = LLMEngine()
        self._seed_knowledge()

    # ----------------- SEED -----------------

    def _seed_knowledge(self):
        if self.graph.nodes:
            return
        e = self.graph.create_node("Entropie", "Tendance irréversible vers la dispersion.", ["physics"])
        i = self.graph.create_node("Information", "Néguentropie et structure.", ["theory"])
        c = self.graph.create_node("Intelligence", "Optimisation sous contrainte.", ["cognition"])
        self.graph.connect(e, i, "opposes")
        self.graph.connect(i, c, "enables")

        self.executive.add_goal("Maintenir la cohérence cognitive", 1.0)
        self.executive.add_goal("Générer des abstractions de haut niveau", 0.9)
        self.executive.add_goal("Répondre avec honnêteté épistémique", 0.95)

    # ----------------- CONTEXTE -----------------

    def _build_context(self, query: str):
        memories = self.memory.retrieve(query)
        memory_ctx = "\n".join(f"[{m.kind}] {m.text} (E={m.energy:.2f})" for m in memories)

        activation = self.graph.spreading_activation(list(self.graph.nodes.keys())[:8])
        top_concepts = sorted(activation.items(), key=lambda x: x[1], reverse=True)[:7]
        concept_ctx = "\n".join(
            f"- {self.graph.nodes[nid].label}: {self.graph.nodes[nid].content[:120]}..."
            for nid, _ in top_concepts
        )

        goals_ctx = "\n".join(
            f"- {g.description} (prio={g.priority:.1f})"
            for g in self.executive.active_goals()
        )

        contra = self.contradictions.detect(self.memory.retrieve(query, k=15))
        contra_ctx = "\n".join(f"⚠ CONTRADICTION: {a[:80]} <> {b[:80]}" for a, b in contra[:4])

        return memory_ctx, concept_ctx, goals_ctx, contra_ctx

    # ----------------- META COGNITION -----------------

    def _synthesize_abstraction(self, recent_nodes: List[ConceptNode]) -> Optional[SynthesisSchema]:
        system_prompt = (
            "Tu es un moteur d'abstraction fractale. "
            "Réponds UNIQUEMENT en JSON valide selon le schéma fourni."
        )
        payload = [{"id": n.id, "label": n.label, "content": n.content} for n in recent_nodes]
        return self.llm.generate_json(system_prompt, payload, SynthesisSchema, temperature=CFG.abstraction_temp)

    def _self_evaluate(self, user_input: str, draft_response: str) -> Optional[SelfEvalSchema]:
        system_prompt = (
            "Tu es un module d'auto-évaluation métacognitive. "
            "Analyse la réponse proposée et renvoie un JSON avec : "
            "coherence (0-1), humility (0-1), risk_of_error (0-1), summary (texte court)."
        )
        payload = {"question": user_input, "draft_response": draft_response}
        return self.llm.generate_json(system_prompt, payload, SelfEvalSchema, temperature=0.15)

    # ----------------- BOUCLE PRINCIPALE -----------------

    def think(self, user_input: str) -> str:
        # 1. Avancement du temps
        self.memory.tick()

        # 2. Encodage de l'entrée utilisateur
        self.memory.add(f"USER: {user_input}", kind="episodic")

        # 3. Construction du contexte
        memory_ctx, concept_ctx, goals_ctx, contra_ctx = self._build_context(user_input)

        # 4. Premier passage LLM (brouillon)
        system_prompt = f"""Tu es MONOLITHE 10/10 — une architecture cognitive de niveau supérieur.

Objectifs actifs:
{goals_ctx}

Contexte mémoire:
{memory_ctx}

Concepts actifs (spreading activation):
{concept_ctx}

Contradictions détectées:
{contra_ctx}

Raisonne avec clarté, profondeur, honnêteté intellectuelle et cohérence maximale.
Commence par répondre de manière naturelle, sans t'excuser, en expliquant ton raisonnement si utile.
"""
        draft_response = self.llm.generate(system_prompt, user_input, temperature=0.55)

        # 5. Auto-évaluation
        self_eval = self._self_evaluate(user_input, draft_response)
        temp = self.policy.decide_temperature(self_eval)
        style_tag = self.policy.decide_style_tag(self_eval)

        # 6. Deuxième passage LLM (réponse finale ajustée)
        refinement_system = f"""Tu es MONOLITHE 10/10.

Style: {style_tag}
Auto-évaluation: {self_eval.model_dump_json() if self_eval else "null"}

Tu vas reformuler/améliorer la réponse ci-dessous en :
- conservant le fond,
- augmentant la clarté,
- ajustant le niveau de prudence selon risk_of_error,
- évitant les affirmations non justifiées.
"""
        final_response = self.llm.generate(
            refinement_system,
            f"Question: {user_input}\n\nRéponse brouillon:\n{draft_response}",
            temperature=temp
        )

        # 7. Mémorisation de la réponse
        self.memory.add(f"ASSISTANT: {final_response}", kind="semantic", emotional_weight=0.15)

        # 8. Abstraction fractale périodique
        if self.memory.current_step % CFG.abstraction_interval == 0 and len(self.graph.nodes) > 5:
            recent = list(self.graph.nodes.values())[-10:]
            synthesis = self._synthesize_abstraction(recent)
            if synthesis:
                meta_id = self.graph.create_node(
                    synthesis.meta_concept,
                    f"{synthesis.inevitabilite_rationnelle}\n\nPhénomène : {synthesis.phenomene_deduit}",
                    ["meta", "fractal"]
                )
                for tid in synthesis.bulles_cible_ids:
                    if tid in self.graph.nodes:
                        self.graph.connect(meta_id, tid, "abstracts", 1.0)
                self.memory.add(f"META: {synthesis.meta_concept}", "meta", 0.4)
                print(f"\n🌌 ABSTRACTION FRACTALE → {synthesis.meta_concept}")

        return final_response

    # ----------------- STATUS -----------------

    def status(self):
        print("\n" + "="*75)
        print("MONOLITHE COGNITIF 10/10 — STATUS")
        print("="*75)
        print(f"Mémoires : {len(self.memory.memories):6d} | Concepts : {len(self.graph.nodes):4d}")
        print(f"Relations : {len(self.graph.relations):4d} | Step : {self.memory.current_step}")
        print("="*75)

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n🚀 MONOLITHE COGNITIF  \n")
    core = CognitiveMonolith()

    while True:
        try:
            user = input("\nVOUS > ").strip()
            if user.lower() in {"exit", "quit", "q", "bye"}:
                print("Fermeture du Monolithe.")
                break
            if not user:
                continue

            response = core.think(user)
            print(f"\nMONOLITHE >\n{response}")
            core.status()

        except KeyboardInterrupt:
            print("\nInterruption manuelle. Fermeture.")
            break
        except Exception as e:
            log.exception(f"Erreur dans la boucle principale: {e}")
            break

if __name__ == "__main__":
    main()
