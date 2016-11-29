# MAC0431 - Introdução a Computação Paralela e Distríbuida

## Integrantes     
* Bruno Endo              7990982
* Danilo Aleixo           7972370
* Gustavo Caparica        7991020


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
   
 
