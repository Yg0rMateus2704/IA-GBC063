# Controle Inteligente de Semáforos com DQN

## Sobre o projeto

O projeto consiste no desenvolvimento de um sistema inteligente de controle de semáforos utilizando Aprendizado por Reforço, por meio do algoritmo Deep Q-Network (DQN).

O objetivo é otimizar o fluxo de veículos em um cruzamento de quatro vias, reduzindo o tamanho das filas e o tempo de espera dos motoristas.

## Como funciona?

O sistema simula o funcionamento de um cruzamento, considerando a chegada de veículos nas quatro direções (Norte, Sul, Leste e Oeste).

O agente DQN observa as condições do trânsito, analisando informações como tamanho das filas, tempo de espera dos veículos, fase atual do semáforo e presença de veículos de emergência.

Com base nessas informações, o agente escolhe entre três ações: manter o sinal verde, trocar a fase do semáforo ou estender o tempo de sinal verde.

A cada decisão, o agente recebe uma recompensa ou penalização de acordo com o resultado de sua ação. Utilizando essas experiências, ele aprende gradualmente a tomar decisões mais eficientes.

O treinamento é realizado em cenários com diferentes intensidades de tráfego, permitindo avaliar o comportamento do agente em situações de fluxo balanceado e desbalanceado.

Por fim, o desempenho do DQN é comparado com um sistema tradicional de semáforos por tempo fixo, permitindo analisar sua eficiência na redução das filas e dos tempos de espera.
