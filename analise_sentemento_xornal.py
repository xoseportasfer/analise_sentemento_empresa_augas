import os
from typing import List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# 1. Definición del Esquema Geopolítico
class ClasificacionNoticia(BaseModel):
    pais_mencionado: str = Field(..., description="El país o región principal de la noticia.")
    tema_principal: str = Field(
        ..., 
        description="Categoría temática de la noticia.",
        enum=["conflicto", "economía", "salud", "clima", "tecnología"]
    )
    tendencia_ideológica: str = Field(
        ..., 
        description="Orientación percibida del texto.",
        enum=["conservadora", "liberal", "neutral", "progresista"]
    )
    resumen_ejecutivo: str = Field(..., description="Breve resumen de máximo 8 palabras.")

# 2. Configuración del Modelo y Parser
llm = ChatOllama(model="mistral", temperature=0)
parser = JsonOutputParser(pydantic_object=ClasificacionNoticia)

# 3. Prompt adaptado a Prensa Geopolítica
tagging_prompt = ChatPromptTemplate.from_template(
    """
    Eres un analista de inteligencia geopolítica. 
    Tu tarea es procesar artículos de prensa y extraer datos estructurados con objetividad clínica.
    
    {format_instructions}
    
    Artículo:
    {input}
    
    Respuesta (JSON):
    """
)

# 4. Matriz de Noticias (Simulacro de diario)
noticias_geopoliticas = [
    "El Congreso de EE.UU. debate un nuevo paquete de ayuda militar para Ucrania en medio del estancamiento del frente oriental.",
    "China anuncia restricciones a la exportación de minerales críticos, golpeando las acciones de empresas tecnológicas en Europa.",
    "La inflación en Argentina alcanza nuevos máximos históricos mientras el gobierno intenta renegociar la deuda con el FMI.",
    "Brote de una nueva variante de gripe en el sudeste asiático pone en alerta a la Organización Mundial de la Salud.",
    "Alemania acelera la construcción de terminales de gas natural para reducir permanentemente la dependencia de los suministros rusos.",
    "El presidente de Brasil propone una moneda común para el Mercosur con el fin de disminuir la hegemonía del dólar.",
    "Ataques con drones en el Mar Rojo interrumpen las rutas comerciales internacionales, elevando el precio del crudo.",
    "Corea del Norte realiza pruebas con misiles balísticos intercontinentales, provocando ejercicios militares conjuntos de Japón y EE.UU.",
    "Inundaciones sin precedentes en Pakistán dejan a millones sin hogar y reabren el debate sobre las reparaciones climáticas globales.",
    "La UE aprueba la primera ley integral para regular la Inteligencia Artificial, buscando liderar la ética digital mundial."
]

# 5. Construcción de la Cadena
# Usamos .partial para inyectar las instrucciones de formato que el parser genera automáticamente
prompt_final = tagging_prompt.partial(format_instructions=parser.get_format_instructions())
tagging_chain = prompt_final | llm | parser

# 6. Procesamiento de la Matriz
resultados_prensa = []

print(f"Analizando {len(noticias_geopoliticas)} noticias geopolíticas...\n")

for i, noticia in enumerate(noticias_geopoliticas, 1):
    try:
        # Invocación de la cadena
        resultado = tagging_chain.invoke({"input": noticia})
        resultados_prensa.append(resultado)
        
        print(f"📊 Noticia #{i}:")
        print(f"   - País: {resultado.get('pais_mencionado')}")
        print(f"   - Tema: {resultado.get('tema_principal')}")
        print(f"   - Tendencia: {resultado.get('tendencia_ideológica')}")
        print(f"   - Resumen: {resultado.get('resumen_ejecutivo')}")
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ Error analizando noticia #{i}: {e}")

print(f"\nAnálisis finalizado. Registros procesados: {len(resultados_prensa)}")