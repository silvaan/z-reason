# TODO

Roteiro para as próximas versões do z-reason e do preprint, em ordem de prioridade dentro de cada bloco.

## Antes de submeter o v1 no arXiv

- [ ] Leitura completa do paper pelos três autores e aval de Thiago e Ivanovitch
- [ ] Confirmar a forma em inglês das afiliações (DCA/UFRN e IFRN Campus Natal-Zona Leste)
- [ ] Verificar o endosso em cs.LG; se preciso, pedir a um coautor já estabelecido na categoria
- [ ] Publicar os checkpoints dos cinco runs como release no GitHub, pois o paper diz que estarão no repositório
- [ ] Submeter com os metadados de `paper/ARXIV.md` e conferir o PDF gerado pelo arXiv

## v2 do preprint: linguagem natural

Este é o principal ponto fraco do v1 e o que mais pesa para uma conferência principal.

### Posicionamento (pesquisa de 30/10/2026)

Desde o v1 surgiram System-One que gastam mais computação raciocinando em texto: GLiDE (Fastino, 1º/10, latência de segundos a minutos e cerca de 10x o custo do Jev), Jeeves (PostHog, chain of thought quando a confiança é menor que 0,9, mal calibrado depois de raciocinar) e o JEVal (12/10), em que destilar o raciocínio para uma passada recupera no máximo 1/4 do ganho. A frase do v1 "não podem usar test-time compute" ficou datada. O nicho aberto é o de passos latentes dentro do orçamento de latência de um System-One. O v2 compete em eficiência e controle de profundidade, não em acurácia geral contra o Jev.

### Estado

- [x] Variable tracking em linguagem natural (`zreason/nl_tasks.py`), com templates de teste reservados
- [x] z-reason do v1 na tarefa em linguagem natural (`train.py --task nl`). Em `runs/nl_s0`, nas profundidades 1–8 com redação vista, o acerto é de 70% com 1 passo e de 100% com 8 ou mais. Com redação nova o modelo fica no chute. Os vetores do Laya também ficam (cerca de 40%). Reservando um grupo de templates por vez, o que quebra é o template de cópia reservado (cai de 100% para 57–60%), e as perguntas reservadas também pesam (72–74%). Com 4 templates de treino por tipo, nenhum encoder aprende a relação independente da redação. Esses runs estão arquivados em `runs/nl_tmpl4`.
- [x] Conjunto maior de templates: 7–10 de treino por tipo de fato e 4 por tipo de pergunta, de 2 a 3 reservados por tipo, e templates de cópia que nomeiam primeiro quem dá o número. As opções do choice não saem mais como "0: None". A fila foi refeita com esse conjunto e baselines de uma passada com `--full-level-by 0.5`.
- [x] Vetores congelados do encoder do Laya no lugar do encoder de caracteres (`--features laya`, `zreason/laya_features.py`)
- [x] ProofWriter OWA com treino em D3 e teste em D5, NatLang e birds-electricity (`--task proofwriter`)
- [x] ProofWriter: com vetores congelados do Laya o modelo decora as 4.816 teorias de treino (treino 1,00, validação ~0,72, `runs/pw_laya_memorized_s0`). Com o encoder de caracteres de 128 posições ele também começa a decorar (validação 0,78). Renomear entidades, atributos e verbos de cada teoria com uma bijeção consistente (`--pw-rename`) resolve: treino e validação andam juntos (0,82 e 0,83 na iteração 5000), e os passos ajudam (0,68 com 1 passo, 0,83 com 8). É uma aumentação só de treino e precisa ser declarada no paper.
- [x] Resultados de 31/10 (24k iterações, renomeação). Em D5 com 16 passos, nas profundidades 1–3 o acerto é 0,96–0,98 nas três seeds, contra 0,67–0,79 do baseline de 1 passo. Na profundidade 4 cai para 0,72–0,88 e na 5 para 0,31–0,41, abaixo da classe majoritária (0,48). A matriz de confusão mostra que nas profundidades 3 a 5 o z-reason quase nunca troca True por False nem erra uma resposta Unknown. Na profundidade 5 ele responde Unknown para cerca de 60% das perguntas True/False, como um provador correto com horizonte limitado. O baseline de 1 passo responde True/False mais vezes na profundidade 5 (0,80), mas erra a maioria das Unknown, o que sugere atalhos estatísticos (citar Zhang et al. 2022, "On the paradox of learning to reason from data"). Em birds-electricity, com vocabulário novo, o z-reason fica em 0,92–0,94 contra 0,80 do baseline de 1 passo. No NatLang todos ficam em ~0,50, porque o encoder de caracteres não lê paráfrases.
- [x] NL com vetores do Laya em 36k iterações (`runs/nl_laya_s0`). Com redação vista: na profundidade 8, 0,56 com 1 passo e 1,00 com 8; na profundidade 16, 0,99; na 24, 0,92. O baseline de 1 passo fica em 0,44–0,47. Com redação nova chega a 0,50–0,63. Na ablação de templates reservados com 16 passos: raiz 0,97, cópia 0,78, pergunta 0,79 e todos 0,60.
- [x] Latência com a GPU ociosa (RTX 3060, 200 repetições, requisição com 4 perguntas). Tarefa NL: 6,7 ms com 1 passo, 17 ms com 8 e 41 ms com 16, mais 52 ms do encoder do Laya. ProofWriter: 14 ms com 1 passo, 30 ms com 8 e 61 ms com 16. Em batch 1 o custo é dominado pelo lançamento de kernels; usar CUDA graphs deixaria mais limpo. As três seeds do NL concordam.
- [ ] Em D5, as profundidades 6 e 7 são 10 perguntas Unknown cuja profundidade é a da prova que falhou. Agrupar em ≥5 nas tabelas.
- [ ] Rodar a fila `run_nl_experiments.sh`: baselines de uma passada (`fixed1`), vetores do Laya, ProofWriter, seeds, `fixed8` e `untied8`
- [ ] Latência por passo, comparada com os cerca de 578 tokens de raciocínio do Jeeves e com o GLiDE
- [ ] Calibração por número de passos no ProofWriter, onde a acurácia não satura
- [x] Rascunho do v2 em `paper/main.tex` (31/10). Tem introdução e related work novos, a seção 6 "From Symbols to Language" (NL, ProofWriter e custo), resultados negativos novos, discussão e conclusão. As tabelas saem de `paper/make_tables.py` e as figuras 7 e 8 de `plots.py`. Os metadados da substituição estão em `paper/ARXIV.md`.
- [ ] Revisão do v2 pelos três autores e submissão como substituição depois que o v1 for anunciado
- [ ] Fine-tuning de um encoder pré-treinado (ModernBERT-base) junto com o núcleo, mais um baseline cross-encoder de uma passada, para o v3

### Resultado negativo para o paper

Usar a cabeça de 2 camadas do Laya como bloco recorrente (`zreason/laya_head.py`, `train_laya.py`) não aprendeu composição em nenhuma das cinco variantes testadas: tokens, fatos, fatos padronizados, frases codificadas separadamente e currículo de tamanho. A sonda mostrou que os vetores de fato do encoder congelado têm cosseno médio de 0,92 entre si. Com padronização o cosseno cai para 0,13, mas o modelo continua sem aprender. A falha da recorrência sobre tokens repete a do v1.

- [x] Versão em linguagem natural da tarefa de variable tracking, com frases variadas no lugar de `a=b` e fatos que precisam ser extraídos do texto
- [x] Trocar o encoder de caracteres por um encoder de sentenças pré-treinado congelado (Laya); falta comparar os dois
- [ ] Segmentar texto livre em fatos, primeiro por sentença e depois com uma alternativa aprendida
- [ ] Uma tarefa de decisão com dados reais no estilo do Jev, como roteamento de tickets ou avaliação por rubrica, que exija combinar mais de um fato
- [ ] Referência externa: um LLM pequeno com saída estruturada, com e sem chain of thought, respondendo as mesmas perguntas, com custo e latência medidos

## Mais tarefas e mais evidência

- [ ] Pelo menos mais uma tarefa algorítmica com controle de dificuldade, como alcançabilidade em grafos, composição de permutações ou a variante com aritmética que já existe em `tasks.py` com `ops=True`
- [ ] Uma tarefa das usadas por Schwarzschild e Bansal, como prefix sums ou labirintos, para comparação direta com a literatura de deep thinking
- [ ] Profundidades acima de 32 com nomes de duas letras, para que o baseline de chute não domine
- [ ] Cinco seeds em vez de três e investigação da falha da seed 2 nas profundidades 28 a 32
- [ ] Testar estabilização da recorrência nessa faixa, com a perda progressiva de Bansal et al. ou a regularização espectral do STARS

## Calibração

- [ ] Tarefa com incerteza irredutível, com rótulos ruidosos ou fatos ambíguos, para medir calibração fora do regime de acerto saturado
- [ ] Reportar Brier e NLL além do ECE, com intervalos por bootstrap
- [ ] Comparar com temperature scaling depois do treino
- [ ] Experimentar um objetivo baseado em resultados, no espírito do RLCD, e comparar com as regras de pontuação próprias

## Parada e custo

- [ ] Cabeça de parada aprendida, no estilo de ACT, PonderNet ou da Q head do TRM, contra a regra de convergência atual
- [ ] Curva de acurácia por custo e medição de latência por passo
- [ ] Codificar o estado uma única vez e compartilhar entre as perguntas; hoje o estado é reprocessado para cada pergunta
- [ ] Revisar a tolerância do `steps="auto"`, que hoje é fixa em 1e-3 e sensível à precisão bfloat16

## Mecanismo

- [ ] Sondas lineares por passo para ver quando o valor de cada variável fica decodificável em cada fato
- [ ] Intervenções causais nas cabeças que olham para descendentes, para testar se são necessárias
- [ ] Repetir a análise de atenção nas três seeds e explicar o crescimento sublinear dos passos necessários

## Baselines

- [ ] Transformer sem compartilhamento de pesos com o mesmo número de parâmetros, e não só com a mesma computação
- [ ] Variantes com 16 e 32 camadas sem compartilhamento
- [ ] Adapter por concatenação do Huginn e recursão no estilo TRM treinados na mesma tarefa

## Interface e escala

- [ ] Conjuntos de resposta maiores, até as 255 opções do Jev, e `score` com 10 níveis
- [ ] Perguntas com instruções em linguagem natural e critérios descritivos longos
- [ ] Modelo maior, para medir se os ganhos com passos se mantêm com mais parâmetros

## Engenharia

- [ ] Empacotar como biblioteca instalável com `pyproject.toml`
- [ ] Servidor HTTP com endpoint compatível com a API do Jev
- [ ] Integração contínua rodando os testes a cada push
- [ ] Checkpoints também no Hugging Face Hub

## Publicação

- [ ] TMLR, submissão contínua, depois de converter para o template da revista
- [ ] Workshops do ICLR 2027, com chamadas previstas entre dezembro de 2026 e fevereiro de 2027
- [ ] BRACIS 2027, com prazo previsto entre abril e maio de 2027, 15 páginas no template Springer e revisão duplo-cega
- [ ] ICML 2027, com prazo no fim de janeiro de 2027, apenas com a versão em linguagem natural pronta
