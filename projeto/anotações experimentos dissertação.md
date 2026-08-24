# Experimentos

## Versão com controle por posição
- new_V1.0:
	- 19/04: 
		- ao analisar graficos da trajetoria da ponta notei que nao estavam batendo com o esperado (ex: posição inicial da ponta nao estava no ponto correto), o problema encontrado foi que ao resetar o estado com env.Reset está não aplicava direto o reset na junta, mas sim aplicava uma ação que mandava o braço pra la (com uma unica ação ele nunca chegava) > isso faz com que os episodios sempre tivessem um residuo do episodio anterior > CORRIGIDO;
		- foi observado que em env.Step os valores dos angulos estavam sendo considerados os valores intencionais e não os valores reais que de fato terminou o braço após a ação > impacto que o agente nao tinha a visão correta do estado fisico > CORRIGIDO;
		- proximos passos:
			- olhar alterações feitas nos 3 files principais > mais detalhes em opencode last /session
			- precisa retreinar;
			- precisa dar eval;
			- precisa coletar episodios para inspeção;
			- precisa rodar todos os scipts de plot;


## Versões com Controle por Torque
- v4.0: rodei com alpha=0.0005 > resultados ruins:

	Pelo seu aggregated_across_seeds.csv, o padrão está bem claro:

	**1) Eixo direct (torque direto) ainda está “difícil”**

		Baseline direct / sem safety filter ficou por volta de ~0.42 de sucesso (médio, mas não alto).

		PI-reward direct (sem safety filter) ficou pior (~0.37) e com maior esforço médio (tau_l1 subiu).

		Safety filter em direct derruba bastante o sucesso (vai pra ~0.30).
		PI+SF em direct dá um ganho pequeno vs SF sozinho, mas ainda fica abaixo do baseline sem SF.

		Interpretação: no torque-control direto, o PI-reward do tipo “penalizar torque” tende a virar trade-off (reduzir esforço / impor prior) e não um “booster de sucesso” automaticamente — principalmente se você já tem lam_a penalizando ação também (muitas vezes fica “penalização dupla de torque”).

	**2) Eixo residual está com “teto” (quase tudo já resolve)**

		Baseline residual já dá ~0.95 sucesso.

		PI-reward e/ou SF em residual mantêm sucesso muito alto, e o SF reduz esforço médio sem destruir performance (bom!).

		Interpretação: residual = “tem PD ajudando” → o problema fica muito mais fácil, então você ganha pouco em sucesso, mas pode mostrar ganhos em métricas de segurança/controle (saturação, intervenção do filtro, delta_tau, esforço, etc.).

	**3) Isso bate com o seu objetivo de texto**

		Você disse: “queremos aumentar sucesso; se esforço maior, citamos como desvantagem”.
		Então, com esses números, a história natural fica:

		Direct: ainda precisa tuning para ficar “baseline forte” antes de tentar provar que PIRL melhora sucesso.

		Residual: é ótimo pra mostrar segurança/eficiência (sem perder sucesso), mas tem pouco espaço pra “melhorar sucesso”.

	**Sobre “antes era position_control e o alpha era 1”**
		Aqui tem uma pegadinha de nomenclatura que explica boa parte da sua dúvida:
			No trecho da sua dissertação de position control, o α=1 é o peso do termo de distância (não é alpha_pi).
			O termo que penaliza torque/PI lá aparece como λτ = 0.0005 (e λ_a = 0.001). Ou seja: o “peso do esforço/torque” lá já era da ordem de 5e-4, que é exatamente o que você está chamando de alpha_pi agora.

			O que mudou de verdade entre “position_control” e “torque_control”:

			Em position_control, a ação era ∆θ e o PD interno garantia rastreio → o sistema fica mais estável e “fácil”, e a penalização de torque pode atuar como regularizador sem impedir movimento.

			Em torque_control, a ação é o torque → se você penaliza torque (PI) e ainda penaliza ação (lam_a), você pode estar desincentivando o próprio ato de se mover. Aí o PI-reward não vira “melhor sucesso”; vira custo extra.
		
- v4.1:
	- foi necessario ajustar o alpha pois:
		- “α foi calibrado para que o termo PI tivesse magnitude comparável ao termo de regularização de ação (ou a uma fração do termo de distância)”;
		- Se PI estava “travando” correções, reduzir alpha pode recuperar success.
		- foram feitos testes com alpha_pi ∈ {0.0, 0.0001, 0.0002, 0.0005} para determinar o que melhora sucesso > a principio vamos seguir com 0.0002, pois:
			1) Resultado principal: qual α deu mais “success” (k=2)
				Success rate por seed (200 episódios):
				α = 1e-4
				seed0: 0.135
				seed1: 0.235
				média k=2: 0.185
				α = 2e-4 ✅ melhor média
				seed0: 0.130
				seed1: 0.360
				média k=2: 0.245
				α = 5e-4
				seed0: 0.060
				seed1: 0.250
				média k=2: 0.155
	- rodada teste comparação com baseline:
		1) Baseline vs PI (train_seed=1, 200 episódios, eval_seed_base=1000)
			**Baseline (direct, pi=0, sf=none)**
			success_rate: 0.275 
			mean_final_dist: 0.509 
			mean_return: -99.19 
			mean_tau_effort: 0.717 
			mean_pi_value: 0.0 

			**PI (direct, pi=1, α=2e-4, sf=none)**

			success_rate: 0.360
			mean_final_dist: 0.366
			mean_return: -78.35
			mean_tau_effort: 0.898
			mean_pi_value: 0.898

		2) Interpretação (direto ao ponto)
			- Ganho em sucesso: 0.360 − 0.275 = +0.085 (8.5 pontos percentuais) no seed1.
			- Melhora de precisão: mean_final_dist caiu bastante (≈ 0.51 → 0.37).
			- Trade-off: esforço subiu (≈ 0.72 → 0.90). Isso é exatamente o tipo de desvantagem que dá pra reportar no texto.
			
	
- v4.2 - somente com baseline e pi-rewards:
	- resultados:
		
			
			
			
			
			
			
			
			

