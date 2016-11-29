


# MAC0431 - Introdução a Computação Paralela e Distríbuida

## Integrantes     
* Bruno Endo              7990982
* Danilo Aleixo           7972370
* Gustavo Caparica        7991020


## Requisitos para fazer o build
```bash
sudo apt-get install nvcc
```

## Resolução do prolema

### CUDA

O problema foi resolvido em CUDA, escolhemos a linguagem pela sua velocidade frente a qualquer outro tipo de solução, ao utilizarmos a perfomance da GPU o programa se manteve consideravelmente mais rápido do que comparado com soluções de colegas que resolveram em linguages para CPU.

### Lógica do programa

* Os códigos que são executados no kernel CUDA, foram divididos em 4 funções para retirar qualquer tipo de interrupção ou condição de dentro do kernel, são elas:
   - 
   - calc_contributions
   - redist
   - re_redist
   
* As bordas tiveram que ser tratadas de maneira especial
   
 
