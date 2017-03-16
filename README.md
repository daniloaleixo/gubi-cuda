# MAC0431 - Introdução a Computação Paralela e Distríbuida

## Integrantes     
* Bruno Endo              7990982
* Danilo Aleixo           7972370
* Gustavo Caparica        7991020

## Problema ## 

Famos fazer um programa para distorcer imagens usando uma interação entre os pixels que obedece a uma física alternativa, muito diferente daquela
que experimentamos no nosso particular universo.
Cada pixel da imagem possui as três componentes usuais: vermelho (R), verde (G) e azul (B). Cada uma possui um comportamento diferente, comandadas
pela componente G.

### Interação entre os pixels ###
As componentes R e B são repelidas pela componente G em direções opostas.
Vamos considerar cada cor como um valor no intervalo [0, 1[, correspondendo à intensidade desta componente no pixel, como usual.
Adicionalmente, o valor de G indica um ângulo, contado a partir do eixo vertical e no sentido horário, com um mapeamento direto para intervalo [0, 2π[, isto é θG = 2πG.
R e B correspondem alinhados com a direção indicada por G, sendo que B aponta na direção oposta. As normas dos vetores correspondem à intensidade de cada componente de cor.
A figura 1 mostra a disposição dos vetores e da componente G, para o pixel (.7, .17, .4).

#### Interações ####
Para simplificar o cálculo, cada pixel interage apenas com seus primeiros vizinhos (acima, abaixo, direita e esquerda). A intensidade da interação
depende das projeções de R e B nos eixos.
<div><img align="left" src="/componentes.png"></div>
Para a componente de cor Cij do pixel (i, j), onde C pode ser R ou B (a cor verde será tratada a seguir), sejam (Cijx, Cijy) suas componentes horizontal e vertical, respectivamente.
Uma parte do valor de Cij será transferida para os vizinhos que se encontram na mesma direção. Por exemplo, se Cijx > 0, parte do seu valor será acrescido a Ci+1,jx, caso contrário, a trasnferência se dará a Ci−1,jx. O procedimento é análogo para a outra componente.
O valor transferido depende do vizinho que o recebe, segundo a seguinte fórmula:
<div><img align="left" src="/equacao.png"></div>
Após todas as transferências, cada pixel deve ser verificado e eventualmente corrigido para que não possua nenhuma cor com valor maior que 1 menor do que 0. Quando um valor passa de 1., o excedente deve ser redistribuído entre os vizinhos uniformemente, desde que isto não provoque novo estouro.
A componente G é atualizada de acordo com os valores obtidos após a atualização de R e B. Considere o vetor (R, B), o ângulo que este vetor forma com a vertical, no sentido horário, deve ser somado a θG e G deve ser corrigido de acordo.
Nota: As bordas são consideradas fixas e nunca se alteram.

### O Programa ###
O programa deve receber o seguinte conjunto de parâmetros de na linha de comando, nesta ordem:
1. Nome do arquivo de entrada
2. Nome do arquivo de saída
3. Número de iterações
4. Número de processadores

### Saída ###
A saída será formada pelo arquivo com a imagem modificada, no formato PPM, tipo P3 (colorido ASCII).

## Requisitos
As dependências do projeto são apenas CUDA (e uma GPU para conseguir rodar).
Assumimos que nvcc se encontra no PATH. Caso contrário será necessário mudar as variáveis NVCC e CUDAPATH dentro do Makefile

## Resolução do prolema

### CUDA

O problema foi resolvido em CUDA, escolhemos a linguagem pela sua velocidade frente a qualquer outro tipo de solução, ao utilizarmos a perfomance da GPU o programa se manteve consideravelmente mais rápido do que comparado com soluções de colegas que resolveram em linguages para CPU.

### Lógica do programa

* Os códigos que são executados no kernel CUDA, foram divididos em 6 funções para retirar qualquer tipo de interrupção ou condição de dentro do kernel, são elas:
   - calc_components: calcula os valores de Rx, Ry e Bx,By para todos os pixels
   - calc_contributions: calcula as contribuições de Rx, Ry e Bx, By para o pixel
   - recalc_magnitudes: recalcula o valor de Rx, Ry e Bx, By a partir das novas componentes
   - redist: se o pixel estoura o valor de 1, distribui o valor para seus vizinhos
   - re_redist (h e w): recoloca os valores que estao na borda "estourada" para dentro da imagem novamemente (fazemos isso para não ter que lidar com condicionais)
   - finalize: limita os valores entre 0 e 1 e calcula G
   
* As bordas tiveram que ser tratadas de maneira especial, para não termos que lidar com condicionais relacionadas à esse problema, colocamos uma linha a mais em cada aresta do quadrado, assim quando a borda real da imagem "estourar", ou seja, ficar maior que 1 e tiver que "espalhar" para seus vizinhos, a nossa borda artificial servirá para receber esse valor, porém depois temos que retirar esses valores da borda artificial e colocá-lo de volta na imagem.
   
 
