# -*- coding: utf-8 -*-
"""
projeto_chatbot.py - Teo, assistente da Entregaí Express (Trabalho P2)
Instalação: pip install spacy requests sqlalchemy
            python -m spacy download pt_core_news_sm
Execução:   python projeto_chatbot.py
"""

import re
import sys
import json
import time
import difflib
import subprocess
import unicodedata
from datetime import datetime, timedelta

import requests
import spacy
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DB_URI = "sqlite:///chatbot_entregas.db"
API = "https://jsonplaceholder.typicode.com"  # API REST de teste
TIMEOUT = 5.0
LIMITE_FALHAS = 2

INICIAL, AGUARDANDO, TRANSBORDO = "INICIAL", "AGUARDANDO_PEDIDO", "TRANSBORDO_HUMANO"
DESCONHECIDA = "intent_desconhecida"
PRIORIDADE = ["reclamar_atraso", "segunda_via_boleto", "rastrear_pedido", "finalizar", "saudacao"]
COM_PEDIDO = {"rastrear_pedido", "segunda_via_boleto", "reclamar_atraso"}
PALAVRAS = {
    "saudacao": {"oi", "oie", "ola", "eai", "bom", "boa"},
    "rastrear_pedido": {"rastrear", "rastreio", "status", "pedido", "entrega", "encomenda"},
    "segunda_via_boleto": {"boleto", "fatura", "pagar", "vencimento"},
    "reclamar_atraso": {"atrasado", "atrasada", "atrasou", "atraso", "demora", "demorou", "reclamar"},
    "finalizar": {"tchau", "adeus", "obrigado", "obrigada", "valeu", "encerrar"},
}
FRASES = {
    "saudacao": ["bom dia", "boa tarde", "boa noite", "e ai"],
    "rastrear_pedido": ["onde esta", "cade meu", "cade o"],
    "segunda_via_boleto": ["segunda via", "2 via", "linha digitavel"],
    "reclamar_atraso": ["nao chegou", "nunca chegou"],
    "finalizar": [],
}
URGENCIA = {"absurdo", "pessimo", "horrivel", "vergonha", "inaceitavel", "ridiculo", "odeio",
            "raiva", "procon", "advogado", "processar", "gerente", "atendente", "humano"}
REGEX_PEDIDO = re.compile(r"(?:#|ped\w*\s*(?:n[o.]?\s*)?#?)\s*(\d{4,10})\b")
REGEX_CPF = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
REGEX_DATA = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
LINHA_BOLETO = "34191.79001 01043.513184 91020.150008 7 90000000024990"  # boleto fictício
HUBS = ["Hub Niterói (Centro)", "Hub São Gonçalo", "Hub Rio de Janeiro (Centro)",
        "Hub Duque de Caxias", "Hub Nova Iguaçu", "Hub Campos dos Goytacazes",
        "Hub Petrópolis", "Hub Macaé", "Hub Volta Redonda", "Hub Cabo Frio"]
FRASES_BOT = {
    "nao_entendi": "Não entendi direito. Você quer rastrear um pedido, pegar a segunda via do boleto ou reclamar de um atraso?",
    "pede_numero": "Não achei o número do pedido aí. Manda só os números, tipo 4821.",
    "lembrete": "Oi de novo! Só falta o número do pedido, pode mandar.",
    "transbordo": "Vou te transferir para um atendente humano agora. Ele já recebe o histórico da conversa, então não precisa explicar tudo de novo. Aguarde um instante.",
    "ja_transbordo": "Seu atendimento já foi encaminhado para um atendente humano. Aguarde, por favor.",
    "tchau": "Obrigado pelo contato! Qualquer coisa é só chamar.",
}
PEDIR_NUMERO = {
    "rastrear_pedido": "Claro! Me passa o número do pedido (ex: #4821) que eu vejo onde ele está.",
    "segunda_via_boleto": "Beleza, vou gerar a segunda via. Qual é o número do pedido?",
    "reclamar_atraso": "Sinto muito pelo atraso. Me informa o número do pedido para eu abrir a reclamação?",
}

Base = declarative_base()


class SessaoUsuario(Base):
    """Estado da conversa e contexto temporário."""
    __tablename__ = "sessoes_usuario"
    session_id = Column(String(50), primary_key=True)
    usuario_id = Column(String(50), nullable=False)
    estado_atual = Column(String(50), nullable=False, default=INICIAL)
    dados_contexto = Column(Text, nullable=False, default="{}")  # JSON
    falhas_consecutivas = Column(Integer, nullable=False, default=0)
    data_criacao = Column(DateTime, default=datetime.now)
    ultima_interacao = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    logs = relationship("ConversaLog", back_populates="sessao", cascade="all, delete-orphan")


class ConversaLog(Base):
    """Histórico de auditoria (LGPD): mensagens, intenções, entidades e eventos."""
    __tablename__ = "logs_conversas"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), ForeignKey("sessoes_usuario.session_id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    remetente = Column(String(20), nullable=False)  # USUARIO ou SISTEMA
    tipo_evento = Column(String(30), nullable=False, default="MENSAGEM")
    mensagem_texto = Column(Text)
    intencao_detectada = Column(String(50))
    entidades_json = Column(Text)
    resposta_texto = Column(Text)
    status_codigo = Column(Integer)
    detalhe_evento = Column(Text)
    sessao = relationship("SessaoUsuario", back_populates="logs")


def normalizar(texto):
    """Minúsculas e sem acento."""
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c)).lower()


class NLUModule:
    """Entende a mensagem: intenção, entidades e urgência."""

    def __init__(self):
        try:
            self.nlp = spacy.load("pt_core_news_sm")
        except OSError:
            subprocess.run([sys.executable, "-m", "spacy", "download", "pt_core_news_sm"], check=True)
            self.nlp = spacy.load("pt_core_news_sm")
        # palavras-chave de cada intenção, para corrigir erros de digitação
        self.vocab = {p: i for i in PRIORIDADE[::-1] for p in PALAVRAS[i]}

    def processar(self, texto):
        doc = self.nlp(texto)
        # texto e lema de cada token (lematização do spaCy)
        cand = {normalizar(t.text) for t in doc if not t.is_space} | \
               {normalizar(t.lemma_) for t in doc if not t.is_space}
        tn = normalizar(texto)

        achadas = set()
        for i in PRIORIDADE:
            if cand & PALAVRAS[i] or any(re.search(rf"\b{f}\b", tn) for f in FRASES[i]):
                achadas.add(i)
        for w in cand:  # erro de digitação: "rastriar" chega perto de "rastrear"
            if len(w) >= 5 and w not in self.vocab:
                perto = difflib.get_close_matches(w, self.vocab, n=1, cutoff=0.8)
                if perto:
                    achadas.add(self.vocab[perto[0]])
        intencao = next((i for i in PRIORIDADE if i in achadas), DESCONHECIDA)

        entidades = {}
        nomes = [e.text for e in doc.ents if e.label_ == "PER"]  # NER: só nomes
        if nomes:
            entidades["NOME"] = nomes
        for nome, regex in [("CPF", REGEX_CPF), ("DATA", REGEX_DATA)]:
            m = regex.search(texto)
            if m:
                entidades[nome] = m.group(0)
        m = REGEX_PEDIDO.search(tn)
        if m:
            entidades["NUMERO_PEDIDO"] = m.group(1)

        termos = sorted(cand & URGENCIA)
        return {"intencao": intencao, "entidades": entidades,
                "urgente": bool(termos) or texto.count("!") >= 3, "termos": termos}


class BusinessLogicModule:
    """Chamadas REST com requests. Nunca deixa a exceção travar o bot."""

    def _req(self, metodo, caminho, **kwargs):
        r = {"metodo": metodo, "url": API + caminho, "http": None,
             "status": "sucesso", "motivo": None, "dados": None}
        inicio = time.time()
        try:
            resp = requests.request(metodo, r["url"], timeout=TIMEOUT, **kwargs)
            r["http"] = resp.status_code
            if resp.status_code in (200, 201):
                r["dados"] = resp.json()
                if not isinstance(r["dados"], dict):
                    raise ValueError("JSON que não é objeto")
            else:
                r["status"], r["motivo"] = "erro_servidor", f"HTTP {resp.status_code}"
        except requests.exceptions.Timeout:
            r["status"], r["motivo"] = "timeout", "o servidor demorou mais de 5 segundos"
        except ValueError:  # JSON inválido (requests.JSONDecodeError também herda de RequestException)
            r["status"], r["motivo"] = "resposta_invalida", "resposta fora do formato JSON"
        except requests.exceptions.RequestException:
            r["status"], r["motivo"] = "falha_conexao", "sem conexão com a API"
        r["tempo_ms"] = (time.time() - inicio) * 1000
        return r

    def rastreio(self, pedido):
        r = self._req("GET", f"/todos/{int(pedido) % 200 or 200}")
        if r["status"] == "sucesso":
            feito = bool(r["dados"].get("completed"))
            r["situacao"] = "Entregue" if feito else "Em rota de entrega"
            r["local"] = HUBS[(r["dados"].get("userId", 1) - 1) % len(HUBS)]
            r["previsao"] = None if feito else "hoje até as 18h"
        return r

    def segunda_via(self, pedido):
        r = self._req("GET", f"/users/{int(pedido) % 10 or 10}")
        if r["status"] == "sucesso":
            valor = 80 + int(pedido) % 400 + 0.90
            r["nome"] = r["dados"].get("name", "Cliente")
            r["valor"] = "R$ " + f"{valor:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
            r["vencimento"] = (datetime.now() + timedelta(days=5)).strftime("%d/%m/%Y")
            r["linha"] = LINHA_BOLETO
        return r

    def reclamacao(self, pedido, texto):
        corpo = {"title": f"Reclamacao de atraso - pedido {pedido}", "body": texto, "userId": 1}
        r = self._req("POST", "/posts", json=corpo)
        r["protocolo"] = f"RC{datetime.now():%y%m%d%H%M%S}"
        return r


class NLGModule:
    """Textos de resposta do bot."""

    def saudacao(self, nome):
        return (f"Oi, {nome.split()[0]}! Eu sou o Teo, assistente da Entregaí Express. "
                "Posso consultar seu pedido, mandar a segunda via do boleto ou registrar "
                "uma reclamação de atraso.")

    def frase(self, chave):
        return FRASES_BOT[chave]

    def pedir(self, intencao):
        return PEDIR_NUMERO[intencao]

    def rastreio(self, pedido, r):
        if r["status"] != "sucesso":
            return f"Não consegui consultar o pedido #{pedido} agora ({r['motivo']}). Tenta de novo daqui a pouco."
        texto = f"Achei o pedido #{pedido}. Status: {r['situacao']}.\nLocal: {r['local']}"
        return texto + (f"\nPrevisão de entrega: {r['previsao']}." if r["previsao"] else "")

    def boleto(self, pedido, r):
        if r["status"] != "sucesso":
            return f"Não consegui gerar o boleto do pedido #{pedido} agora ({r['motivo']}). Tenta de novo em alguns minutos."
        return (f"Segunda via do pedido #{pedido} gerada.\nSacado: {r['nome']}\nValor: {r['valor']}\n"
                f"Vencimento: {r['vencimento']}\nLinha digitável: {r['linha']}")

    def reclamacao(self, pedido, protocolo):
        return (f"Sinto muito pelo atraso do pedido #{pedido}. Sua reclamação foi registrada com o "
                f"protocolo {protocolo} e segue como prioridade para a equipe de logística.")


class DatabaseModule:
    """SQLAlchemy + SQLite: sessão e auditoria."""

    def __init__(self, uri=DB_URI):
        self.engine = create_engine(uri)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    @staticmethod
    def mascarar(texto):
        """LGPD: CPF gravado sem os primeiros dígitos."""
        if not isinstance(texto, str):
            return texto
        return REGEX_CPF.sub(lambda m: "***.***.***-" + re.sub(r"\D", "", m.group(0))[-2:], texto)

    def obter_sessao(self, sid, uid):
        with self.Session() as db:
            s = db.get(SessaoUsuario, sid)
            if s is None:
                s = SessaoUsuario(session_id=sid, usuario_id=uid, estado_atual=INICIAL, dados_contexto="{}")
                db.add(s)
                db.commit()
            return {"estado": s.estado_atual, "falhas": s.falhas_consecutivas,
                    "contexto": json.loads(s.dados_contexto or "{}")}

    def salvar_turno(self, sid, estado, falhas, ctx, msg, intencao, entidades, resposta, status, eventos):
        """Grava o turno inteiro numa transação: mensagem, resposta e eventos."""
        with self.Session() as db:
            s = db.get(SessaoUsuario, sid)
            s.estado_atual, s.falhas_consecutivas = estado, falhas
            s.dados_contexto = json.dumps(ctx, ensure_ascii=False)
            ents = {k: self.mascarar(v) for k, v in entidades.items()}
            db.add(ConversaLog(session_id=sid, remetente="USUARIO", tipo_evento="MENSAGEM",
                               mensagem_texto=self.mascarar(msg), intencao_detectada=intencao,
                               entidades_json=json.dumps(ents, ensure_ascii=False),
                               resposta_texto=resposta, status_codigo=status))
            for tipo, detalhe, st in eventos:
                db.add(ConversaLog(session_id=sid, remetente="SISTEMA", tipo_evento=tipo,
                                   detalhe_evento=self.mascarar(detalhe), status_codigo=st))
            db.commit()


class ProjetoChatbot:
    """Chassi: máquina de estados, fallback e transbordo."""

    def __init__(self, db_uri=DB_URI):
        print("=== ASSISTENTE TEO (ENTREGAÍ EXPRESS) ===")
        self.nlu = NLUModule()
        self.api = BusinessLogicModule()
        self.nlg = NLGModule()
        self.db = DatabaseModule(db_uri)

    def _transbordar(self, ctx, eventos, motivo):
        ctx["motivo_transbordo"] = motivo
        eventos.append(("TRANSBORDO", f"Encaminhado para atendente humano. Motivo: {motivo}", 200))
        return self.nlg.frase("transbordo")

    def _atender(self, intencao, pedido, texto, eventos):
        """Chama a API certa e devolve (resposta, status)."""
        if intencao == "reclamar_atraso":
            r = self.api.reclamacao(pedido, texto)
            eventos.append(("API", self._det_api(r), r["http"]))
            eventos.append(("RECLAMACAO", f"Pedido {pedido} | protocolo {r['protocolo']} | texto: {texto}", 200))
            return self.nlg.reclamacao(pedido, r["protocolo"]), 200
        if intencao == "rastrear_pedido":
            r, texto_resp = self.api.rastreio(pedido), self.nlg.rastreio
        else:
            r, texto_resp = self.api.segunda_via(pedido), self.nlg.boleto
        eventos.append(("API", self._det_api(r), r["http"]))
        return texto_resp(pedido, r), (200 if r["status"] == "sucesso" else 503)

    @staticmethod
    def _det_api(r):
        return f"{r['metodo']} {r['url']} -> {r['http'] or r['status']} em {r['tempo_ms']:.0f} ms"

    def responder(self, sid, uid, msg):
        a = self.nlu.processar(msg)
        intencao, ents = a["intencao"], a["entidades"]
        s = self.db.obter_sessao(sid, uid)
        estado, falhas, ctx = s["estado"], s["falhas"], s["contexto"]
        eventos, status = [], 200

        # número solto ("4821") só vale quando o bot pediu o número
        pedido = ents.get("NUMERO_PEDIDO")
        if not pedido and estado == AGUARDANDO and re.fullmatch(r"#?\s*\d{4,10}", msg.strip()):
            pedido = msg.strip().lstrip("#").strip()
            ents["NUMERO_PEDIDO"] = pedido
        if estado == AGUARDANDO and pedido and intencao == DESCONHECIDA:
            intencao = ctx.get("pendente", intencao)

        if estado == TRANSBORDO:
            resposta = self.nlg.frase("ja_transbordo")
        elif a["urgente"]:
            resposta = self._transbordar(ctx, eventos, "insatisfação/urgência: " + (", ".join(a["termos"]) or "pontuação forte"))
            estado = TRANSBORDO
        elif intencao == DESCONHECIDA:
            falhas += 1
            if falhas >= LIMITE_FALHAS:
                resposta = self._transbordar(ctx, eventos, f"{falhas} falhas seguidas de entendimento")
                estado = TRANSBORDO
            else:
                resposta = self.nlg.frase("pede_numero" if estado == AGUARDANDO else "nao_entendi")
        else:
            falhas = 0
            if intencao in COM_PEDIDO and pedido:
                resposta, status = self._atender(intencao, pedido, msg, eventos)
                ctx.pop("pendente", None)
                ctx["ultimo_pedido"] = pedido
                estado = INICIAL
            elif intencao in COM_PEDIDO:
                ctx["pendente"], estado = intencao, AGUARDANDO
                resposta = self.nlg.pedir(intencao)
            elif intencao == "finalizar":
                resposta, estado = self.nlg.frase("tchau"), INICIAL
                ctx.pop("pendente", None)
            else:  # saudação
                resposta = self.nlg.frase("lembrete") if estado == AGUARDANDO else self.nlg.saudacao(uid)

        self.db.salvar_turno(sid, estado, falhas, ctx, msg, intencao, ents, resposta, status, eventos)
        return resposta


def extrair_nome(texto):
    """Pega só o nome, mesmo se a pessoa digitar uma frase."""
    m = re.search(r"(?:meu nome (?:é|e)|me chamo|sou o|sou a)\s+([A-Za-zÀ-ú]+)", texto, re.I)
    if m:
        return m.group(1).capitalize()
    ignorar = {"oi", "oie", "ola", "olá", "eai", "bom", "boa", "dia", "tarde", "noite", "meu",
               "nome", "e", "é", "sou", "eu", "me", "chamo", "tudo", "bem"}
    palavras = [w for w in re.findall(r"[A-Za-zÀ-ú]+", texto) if w.lower() not in ignorar]
    return palavras[0].capitalize() if palavras else "Visitante"


if __name__ == "__main__":
    bot = ProjetoChatbot()
    sid = f"CHAT_{datetime.now():%Y%m%d%H%M%S}"
    try:
        nome = extrair_nome(input("Seu nome: ").strip())
    except (KeyboardInterrupt, EOFError):
        nome = "Visitante"
    print(f"Teo: {bot.nlg.saudacao(nome)}")
    while True:
        try:
            msg = input("Você: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not msg:
            continue
        if msg.lower() in ("sair", "exit"):
            print(f"Teo: {bot.nlg.frase('tchau')}")
            break
        if msg.lower() == "novo":
            sid = f"CHAT_{datetime.now():%Y%m%d%H%M%S}"
            print("Teo: Certo, novo atendimento iniciado.\n")
            continue
        print(f"Teo: {bot.responder(sid, nome, msg)}\n")
