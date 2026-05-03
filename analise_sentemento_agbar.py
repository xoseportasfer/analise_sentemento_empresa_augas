import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

llm = ChatOllama(model="mistral", temperature=0)

#llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

tagging_prompt = ChatPromptTemplate.from_template(
    """
    Eres un asistente analista de soporte técnico.
    Extrae la información del siguiente ticket de cliente y responde ÚNICAMENTE en formato JSON siguiendo el esquema proporcionado.

    Tienes que realizar un análisis de sentimiento del cliente, identificar el departamento al que debe ser derivado el caso, asignar un nivel de prioridad basado en la gravedad del reporte y detectar el idioma en el que está redactado el ticket.
    
    Además tienes que generar una breve descripción de máximo 6 palabras sobre el motivo de la consulta, que se utilizará para clasificar el ticket internamente.

    Ticket:
    {input}
    
    Respuesta (JSON):
    """
)

tickets_soporte_agua = [
    "Hola, he recibido mi factura de este mes y el importe es el triple de lo habitual sin haber cambiado mi consumo. Por favor, revisen si hay un error de lectura.",
    "¡Es una vergüenza! Llevo toda la mañana sin una gota de agua en casa y nadie nos ha avisado de cortes en la zona. Exijo una solución inmediata.",
    "Buenos días, me gustaría solicitar información sobre las tarifas comerciales para abrir un nuevo local de restauración en el centro.",
    "Tengo una pequeña fuga en la llave de paso de mi jardín. No es urgente, pero me gustaría que un operario viniera a echarle un vistazo la semana que viene.",
    "Estimados, quiero domiciliar mis recibos en una cuenta bancaria diferente. ¿Podrían indicarme qué formulario debo rellenar?",
    "Llevo tres días viendo cómo sale agua de una tubería rota en la acera frente a mi portal. Es un desperdicio enorme y nadie viene a arreglarlo.",
    "He intentado darme de alta en el portal del cliente pero la página me da error 404 constantemente. Necesito descargar mi última factura.",
    "Quisiera dar de baja el suministro de mi antigua vivienda ya que el contrato de alquiler finaliza este viernes.",
    "El agua que sale del grifo tiene un color amarillento y un olor metálico muy fuerte desde hace dos horas. Me preocupa que no sea potable.",
    "Gracias por la rapidez en la gestión anterior, pero sigo esperando que me confirmen si el descuento por familia numerosa ya está aplicado."
]

from typing import Optional

class ClasificacionTicket(BaseModel):
    """Información extraída de un ticket de soporte de la compañía de agua."""
    
    prioridad: str = Field(
        ...,
        description="Nivel de urgencia. Fugas externas, falta de suministro o problemas de salud son 'Alta'.",
        enum=["Baja", "Media", "Alta"]
    )
    
    sentimiento: str = Field(
        ...,
        description="El tono emocional del mensaje del cliente.",
        enum=["Agradecido", "Neutral", "Molesto", "Enojado"]
    )
    
    idioma: str = Field(
        ..., 
        description="El idioma en el que está escrito el ticket.",
        enum=["español", "inglés", "otro"]
    )

    descripcion_breve: str = Field(
        ...,
        description="Un resumen de máximo 6 palabras sobre el motivo de la consulta."
    )

    departamento: str = Field(
        ...,
        description="El departamento al que debe derivarse el caso",
        enum=["ventas", "técnico", "facturación"]
    )

# 2. Creamos el parser basado en tu clase
parser = JsonOutputParser(pydantic_object=ClasificacionTicket)

# LLM
llm = ChatOllama(model="mistral", temperature=0)

""".with_structured_output(
    ClasificacionTicket
)"""

llm_estructurado = llm

# 4. Inyectamos las instrucciones del parser en el prompt
prompt_final = tagging_prompt.partial(format_instructions=parser.get_format_instructions())

# Luego creas la cadena uniendo el prompt con ese modelo ya preparado
tagging_chain = tagging_prompt | llm_estructurado | parser


# 1. Definición de la matriz de resultados
resultado_sentimiento = []

# 2. Configuración de la cadena (Chain)
chain_tickets = tagging_prompt | llm

print(f"Iniciando el procesamiento de {len(tickets_soporte_agua)} tickets...\n")

# 3. Bucle para tratar los tickets



for i, ticket in enumerate(tickets_soporte_agua, 1):
    try:
        resultado = tagging_chain.invoke({"input": ticket})
        
        if resultado:
            resultado_sentimiento.append(resultado)
            # Acceso como diccionario
            print(f"✅ Ticket #{i}:")
            print(f"   - Dept: {resultado.get('department', 'N/A')}")
            print(f"   - Prioridad: {resultado.get('priority', 'N/A')}")
            print(f"   - Sentimiento: {resultado.get('sentiment', 'N/A')}")
            print(f"   - Lenguaje: {resultado.get('language', 'N/A')}")
        else:
            print(f"⚠️ Ticket #{i}: El parser no pudo extraer datos.")

    except Exception as e:
        print(f"❌ Error en Ticket #{i}: {e}")





# 4. Verificación final
print(f"\nProcesamiento completado. Total de registros en la matriz: {len(resultado_sentimiento)}")





