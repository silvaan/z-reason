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

- [ ] Versão em linguagem natural da tarefa de variable tracking, com frases variadas no lugar de `a=b` e fatos que precisam ser extraídos do texto
- [ ] Trocar o encoder de caracteres por um encoder de sentenças pré-treinado, congelado ou com ajuste leve, e comparar os dois
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
