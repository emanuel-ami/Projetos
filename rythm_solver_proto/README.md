# Rhythm Solver Bot

Bot de detecção e automação de teclas para jogos de ritmo com **otimizações avançadas**.

## ⚡ Otimizações Implementadas

- **✨ Template Matching Paralelo**: Processa todas as direções simultaneamente usando ThreadPoolExecutor (4 workers)
- **🎯 Timing Preditivo**: Calcula velocidade das setas e pressiona no momento ideal
- **🚀 Teclas Não-Bloqueantes**: Key presses assíncronos não atrasam o loop principal
- **📊 Compensação de Delay**: Estima e compensa automaticamente o delay de processamento
- **🔧 Auto-Resize de Templates**: Redimensiona templates grandes automaticamente

## Requisitos

- Python 3.13+
- Bibliotecas instaladas (veja `requirements.txt`)
- **Executar como Administrador** (necessário para enviar teclas)

## Como Usar

### 1. Primeira Execução

```powershell
& C:/Python313/python.exe ryth_solver.py
```

Na primeira execução, você precisará calibrar duas regiões:

1. **Arrow Region (Verde)**: Área onde as setas aparecem
   - Arraste o mouse para desenhar um retângulo ao redor da área de detecção
   - Clique para confirmar

2. **HIT Zone (Vermelho)**: Zona onde as setas devem ser pressionadas
   - Desenhe um retângulo na linha/zona de hit
   - Clique para confirmar

As regiões são salvas em `regions.json` e carregadas automaticamente nas próximas execuções.

### 2. Controles Durante Execução

| Tecla | Ação |
|-------|------|
| **R** | Recalibrar Arrow Region (área de detecção) |
| **/** | Recalibrar HIT Zone (zona de acionamento) |
| **[** | Diminuir escala dos templates (0.1 por vez) |
| **]** | Aumentar escala dos templates (0.1 por vez) |
| **ESC** | Sair do programa |

### 3. Templates de Setas

O bot procura por estas imagens na pasta do projeto:
- `up.png` → Pressiona **W**
- `down.png` → Pressiona **S**
- `left.png` → Pressiona **A**
- `right.png` → Pressiona **D**

**Como criar os templates:**
1. Tire uma screenshot do jogo
2. Recorte apenas UMA seta (sem fundo extra)
3. Salve como PNG em escala de cinza (opcional, o script converte automaticamente)

### 4. Ajustes de Precisão

Se as detecções não estiverem boas:

- **Templates muito grandes/pequenos**: Use `[` e `]` para ajustar a escala
- **Muitos falsos positivos**: Aumente `THRESHOLD` no código (linha 13)
- **Poucos hits**: Diminua `THRESHOLD` ou recalibre as regiões

## Troubleshooting

### Erro: "gdi32.GetDIBits() failed"
- Causa: Região inválida (largura ou altura = 0)
- Solução: Delete `regions.json` e recalibre pressionando **R** e **/**

### Teclas não são pressionadas
- Execute o VS Code/terminal como **Administrador**
- Certifique-se de que a janela do jogo está em foco (primeiro plano)
- Alguns jogos com anti-cheat podem bloquear teclas simuladas

### Templates não detectados
- Verifique se os arquivos PNG existem na pasta
- Use `[` e `]` para ajustar a escala dos templates
- Certifique-se de que as imagens são recortes limpos das setas

## Configuração Avançada

Edite estas variáveis no início do script:

### Configurações Básicas
```python
THRESHOLD = 0.82         # Sensibilidade de detecção (0.0 a 1.0) - aumentado para menos falsos positivos
template_scale = 1.0     # Escala inicial dos templates
KEY_MAP = {              # Mapeamento de teclas
    "up": "w",
    "down": "s",
    "left": "a",
    "right": "d"
}
```

### Configurações de Otimização
```python
MAX_WORKERS = 4                # Threads para template matching paralelo (ajuste conforme CPU)
MAX_TEMPLATE_SCALE = 0.25      # Máximo 25% do tamanho do frame (evita templates muito grandes)
HIT_OFFSET_COMP_MS = 0.0       # Compensação manual de timing em ms (positivo = pressionar mais cedo)
PREDICT_WINDOW = 5             # Número de detecções para calcular velocidade
MIN_VELOCITY_PIX_PER_SEC = 1   # Velocidade mínima para ativar predição
SHOW_DEBUG = True              # Mostrar janela de debug (False para máximo desempenho)
```

### Como Ajustar o Timing

Se o bot estiver pressionando **muito cedo**:
- Diminua `HIT_OFFSET_COMP_MS` (valores negativos atrasam)
- Aumente `THRESHOLD` para detectar mais tarde

Se o bot estiver pressionando **muito tarde**:
- Aumente `HIT_OFFSET_COMP_MS` (ex: 50.0 para 50ms mais cedo)
- Diminua `THRESHOLD` para detectar mais cedo

### Desempenho

Para **máximo FPS**:
```python
SHOW_DEBUG = False       # Desabilita janela (economiza ~30-50% CPU)
MAX_WORKERS = 6          # Se tiver CPU com 6+ cores
```

Para **máxima precisão**:
```python
THRESHOLD = 0.85         # Menos falsos positivos
PREDICT_WINDOW = 7       # Mais histórico para cálculo de velocidade
```

## Notas

- **Template Matching Paralelo**: Detecta todas as 4 direções ao mesmo tempo (4x mais rápido)
- **Timing Preditivo**: Analisa velocidade das setas e pressiona no momento ideal (não apenas quando overlap)
- **Compensação Automática**: O bot estima seu próprio delay de processamento e compensa
- **Teclas Assíncronas**: Pressionar teclas não bloqueia o loop de detecção
- A taxa de detecção depende da CPU (processamento de cada frame)
- Funciona melhor com jogos em janela ou borderless
- O debug overlay mostra: escala, threshold, delay de processamento em ms, e informações de predição
