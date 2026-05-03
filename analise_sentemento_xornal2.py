import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# 1. Definición del Esquema Completo
class ClasificacionNoticia(BaseModel):
    pais_mencionado: str = Field(..., description="El país o región principal de la noticia.")
    
    tema_principal: str = Field(
        ..., 
        description="Categoría temática de la noticia.",
        enum=["conflicto", "economía", "salud", "clima", "tecnología", "política"]
    )
    
    tendencia_ideológica: str = Field(
        ..., 
        description="Orientación percibida del texto.",
        enum=["conservadora", "liberal", "neutral", "progresista"]
    )
    
    sentimiento: str = Field(
        ...,
        description="La carga emocional del texto.",
        enum=["positivo", "negativo", "neutral", "muy crítico"]
    )
    
    resumen_ejecutivo: str = Field(..., description="Breve resumen de máximo 8 palabras.")

# 2. Configuración del Modelo y Parser
llm = ChatOllama(model="mistral", temperature=0)
parser = JsonOutputParser(pydantic_object=ClasificacionNoticia)

# 3. Prompt con todas las variables
tagging_prompt = ChatPromptTemplate.from_template(
    """
    Eres un analista de inteligencia geopolítica. 
    Tu tarea es procesar el artículo y extraer datos estructurados sobre el tema, tendencia y sentimiento.
    Si el autor usa lenguaje sesgado, identifícalo claramente en las variables correspondientes.

    {format_instructions}
    
    Artículo:
    {input}
    
    Respuesta (JSON):
    """
)

# 4. Matriz con Alta Variabilidad (Especial para forzar sesgos)
noticias_geopoliticas = [
    "La resistencia heroica de las milicias populares logra frenar el avance del imperialismo en la frontera, una victoria para la soberanía.",
    "El nuevo paquete de austeridad es una medida valiente y necesaria para salvar la libertad económica frente al populismo asfixiante.",
    "La cumbre climática termina en un fracaso absoluto por la avaricia de las potencias industriales que prefieren el lucro al planeta.",
    "Un informe revela que las políticas de fronteras abiertas están destruyendo la identidad cultural y colapsando los servicios públicos.",
    "El gobierno celebra el éxito de la reforma agraria que hace justicia social con los campesinos históricamente oprimidos.",
    "La dictadura expansionista del país vecino amenaza la paz regional con una retórica bélica que recuerda a los capítulos más oscuros.",
    "El libre mercado ha demostrado ser el único motor capaz de sacar a millones de la pobreza, pese a las trabas estatales.",
    "Activistas denuncian que la nueva ley de seguridad es una herramienta represiva que busca silenciar la disidencia política.",
    "La alianza estratégica entre potencias emergentes representa un nuevo mundo multipolar que desafía la hegemonía de Occidente.",
    "Inversores internacionales aplauden la desregulación total del sector minero como el único camino hacia el progreso real."
]

# 4. Inyección de instrucciones y creación de la cadena
prompt_final = tagging_prompt.partial(format_instructions=parser.get_format_instructions())
tagging_chain = prompt_final | llm | parser

# 5. Procesamiento de la matriz
resultado_analisis = []

print(f"Iniciando análisis geopolítico completo...\n")

for i, noticia in enumerate(noticias_geopoliticas, 1):
    try:
        resultado = tagging_chain.invoke({"input": noticia})
        resultado_analisis.append(resultado)
        
        print(f"📊 NOTICIA #{i}")
        print(f"   - País: {resultado.get('pais_mencionado')}")
        print(f"   - Tema: {resultado.get('tema_principal')}")  # Variable restaurada
        print(f"   - Tendencia: {resultado.get('tendencia_ideológica')}")
        print(f"   - Sentimiento: {resultado.get('sentimiento')}")
        print(f"   - Resumen: {resultado.get('resumen_ejecutivo')}")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error en Noticia #{i}: {e}")

print(f"\nProcesamiento completado. Total en matriz: {len(resultado_analisis)}")