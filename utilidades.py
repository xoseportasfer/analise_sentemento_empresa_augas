for i, ticket in enumerate(tickets_soporte_agua, 1):
    print(f"Procesando Ticket #{i}...")
    
    try:
        # Invocamos la cadena para el ticket actual
        resultado = chain_tickets.invoke({"input": ticket})
        
        # Añadimos el objeto extraído a nuestra matriz de resultados
        resultado_sentimiento.append(resultado)
        
        # Feedback visual por consola
        print(f"✅ Ticket #{i}:")

        print(f"   - Dept: {resultado['departamento']}")
        print(f"   - Prioridad: {resultado['prioridad']}")
        print(f"   - Sentimiento: {resultado['sentimiento']}")
        print(f"   - Resumen: {resultado['descripcion_breve']}")

    except Exception as e:
        print(f"   ❌ Error al procesar el ticket #{i}: {e}")
        # Opcional: añadir un valor nulo para mantener la correlación de índices
        resultado_sentimiento.append(None)
    
    print("-" * 30)

class Classification(BaseModel):
    """Clasificación técnica y emocional de un ticket de soporte de agua."""

    sentiment: str = Field(
        ..., 
        description="El tono emocional del mensaje del cliente",
        enum=["feliz", "neutral", "molesto", "enojado"]
    )
    
    department: str = Field(
        ...,
        description="El departamento al que debe derivarse el caso",
        enum=["ventas", "técnico", "facturación"]
    )
    
    priority: str = Field(
        ...,
        description="Nivel de urgencia basado en la gravedad del reporte",
        enum=["baja", "media", "alta"]
    )
    
    language: str = Field(
        ..., 
        description="Idioma en el que está redactado el ticket",
        enum=["spanish", "english"]
    )
    
tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)


llm = ChatOllama(model="mistral", temperature=0).with_structured_output(
    Classification
)