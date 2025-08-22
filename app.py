import streamlit as st
import json
import pandas as pd
import psycopg2 
from decimal import Decimal
import google.generativeai as genai
import re
from supabase import create_client, Client

# Carregar credenciais de forma segura
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_api_key)
except KeyError:
    st.error("Chave 'GEMINI_API_KEY' não encontrada no secrets.toml.")
    st.stop()

supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]

# Credenciais do PostgreSQL
pg_host = st.secrets["PG_HOST"]
pg_port = st.secrets["PG_PORT"]
pg_database = st.secrets["PG_DATABASE"]
pg_user = st.secrets["PG_USER"]
pg_password = st.secrets["PG_PASSWORD"]

# Inicializar os clientes
model = genai.GenerativeModel('gemini-1.5-flash-latest') 
supabase_client: Client = create_client(supabase_url, supabase_key)

st.title("Chatbot OPS")
st.write("Pergunte sobre beneficiários, prestadores, autorizações, mensalidades e contas.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

@st.cache_data
def get_database_schema():
    """Recupera e formata o dicionário de dados do Supabase."""
    
    tabelas_data = supabase_client.from_("omni_dic_tabela").select("nm_tabela, ds_tabela").execute().data
    tabelas_df = pd.DataFrame(tabelas_data)
    
    atributos_data = supabase_client.from_("omni_dic_atributo").select("nm_tabela, nm_atributo, ds_tipo_dado, cd_dominio, ds_atributo").execute().data
    atributos_df = pd.DataFrame(atributos_data)
    
    constraints_data = supabase_client.from_("omni_dic_constraint").select("ds_tipo_constraint, nm_tabela, nm_atributo, nm_tabela_referenciada, nm_atributo_referenciado").execute().data
    constraints_df = pd.DataFrame(constraints_data)
    
    dominios_data = supabase_client.from_("omni_dic_dominio").select("cd_dominio, ds_dominio").execute().data
    dominios_df = pd.DataFrame(dominios_data)

    # NOVO: Recuperar dados das novas tabelas
    conceito_data = supabase_client.from_("omni_dic_conceito").select("nm_conceito, ds_conceito").execute().data
    sql_exemplo_data = supabase_client.from_("omni_dic_sql_exemplo").select("ds_metrica, ds_sql").execute().data

    schema_info = "## Estrutura do Banco de Dados\n\n"
    schema_info += "### Tabelas e Descrições\n"
    for _, row in tabelas_df.iterrows():
        schema_info += f"- **{row['nm_tabela']}**: {row['ds_tabela']}\n"
    
    schema_info += "\n### Atributos das Tabelas\n"
    for tabela in atributos_df['nm_tabela'].unique():
        schema_info += f"#### {tabela}\n"
        tabela_atributos = atributos_df[atributos_df['nm_tabela'] == tabela]
        for _, row in tabela_atributos.iterrows():
            schema_info += f"- **{row['nm_atributo']}** ({row['ds_tipo_dado']}): {row['ds_atributo']}\n"
            if pd.notna(row['cd_dominio']) and row['cd_dominio']:
                dominios = dominios_df[dominios_df['cd_dominio'] == row['cd_dominio']]
                if not dominios.empty:
                    valores = ', '.join([f"'{d}'" for d in dominios['ds_dominio'].tolist()])
                    schema_info += f"  (Valores possíveis: {valores})\n"

    schema_info += "\n### Relacionamentos entre Tabelas\n"
    for _, row in constraints_df.iterrows():
        if row['ds_tipo_constraint'] == 'Foreign Key':
            schema_info += (f"- `{row['nm_tabela']}.{row['nm_atributo']}` referencia "
                            f"`{row['nm_tabela_referenciada']}.{row['nm_atributo_referenciado']}`\n")
                            
    # NOVO: Adicionar conceitos de negócio
    schema_info += "\n## Conceitos de Negócio\n"
    for item in conceito_data:
        schema_info += f"- **{item['nm_conceito']}**: {item['ds_conceito']}\n"
        
    # NOVO: Adicionar exemplos de consultas
    schema_info += "\n## Exemplos de Consultas\n"
    for item in sql_exemplo_data:
        schema_info += f"- Pergunta: {item['ds_metrica']}\n  SQL: {item['ds_sql']}\n"

    return schema_info

database_schema = get_database_schema()

def get_sql_query_from_llm(prompt):
    """Gera a consulta SQL baseada na pergunta do usuário."""
    sql_prompt = (
        f"Você é um especialista em SQL. Sua tarefa é converter a pergunta do usuário em uma "
        f"consulta SQL válida, **usando estritamente APENAS** as tabelas e colunas fornecidas no esquema. "
        f"Retorne APENAS a consulta SQL, sem nenhuma outra explicação ou formatação de código.\n\n"
        f"Regras:\n"
        f"1. Use os nomes de tabelas e atributos exatamente como estão no esquema, em minúsculo.\n"
        f"2. Para comparações de strings (cláusula WHERE), utilize a função `UPPER(unaccent())` em ambos os lados para garantir que a comparação seja insensível a maiúsculas/minúsculas e a acentos.\n"
        f"3. NUNCA utilize tabelas, colunas, ou entidades que não estejam no esquema fornecido.\n"
        f"4. Quando um atributo possuir valores possíveis (domínios), utilize-os para filtros na cláusula WHERE.\n"
        f"5. Use JOINs apenas para tabelas que estejam explicitamente relacionadas no esquema.\n"
        f"6. Para contagem, utilize `COUNT(*)` ou `COUNT(1)`.\n"
        f"7. Para calcular a média de uma contagem ou soma (agregação aninhada), use uma subquery. Por exemplo, `SELECT AVG(total_contas) FROM (SELECT COUNT(*) AS total_contas FROM CONTA GROUP BY ID_PRESTADOR_ENVIO) AS subquery;`.\n\n"
        f"Esquema do Banco de Dados:\n{database_schema}\n\n"
        f"Pergunta do usuário: {prompt}"
    )
    
    response = model.generate_content(sql_prompt, generation_config=genai.types.GenerationConfig(
        temperature=0.1,
    ))
    return response.text.strip()

def execute_sql_query(query):
    """Executa a consulta SQL usando psycopg2 e retorna os dados."""
    conn = None
    data = None
    try:
        query = re.sub(r'^\s*```sql|```\s*$', '', query, flags=re.MULTILINE).strip()
        
        conn = psycopg2.connect(
            host=pg_host,
            port=pg_port, 
            database=pg_database,
            user=pg_user,
            password=pg_password
        )
        cur = conn.cursor()
        cur.execute(query)
        
        if cur.description:
            data = cur.fetchall()
            column_names = [desc[0] for desc in cur.description]
            
            results_as_dict = [dict(zip(column_names, row)) for row in data]
            
            for row_dict in results_as_dict:
                for key, value in row_dict.items():
                    if isinstance(value, Decimal):
                        row_dict[key] = float(value)
        else:
            results_as_dict = {"message": f"{cur.rowcount} linhas afetadas."}

        cur.close()
        conn.close()
        return results_as_dict
    except Exception as e:
        st.error(f"Erro ao executar a consulta SQL: {e}")
        if conn:
            conn.close()
        return None

def get_final_response_from_llm(prompt, data):
    """Gera a resposta final usando a pergunta original e os dados recuperados."""
    final_prompt = (
        f"Baseado na seguinte pergunta do usuário e nos dados obtidos do banco de dados, "
        f"gere uma resposta completa, clara e amigável. Os dados estão em formato JSON.\n\n"
        f"na resposta, não coloque informações ténicas do banco de dados ou das tabelas, "
        f"responda como se estivesse falando com o usuário final.\n\n"
        f"Pergunta do usuário: {prompt}\n"
        f"Esquema do Banco de Dados:\n{database_schema}\n\n"
        f"Dados do banco de dados: {json.dumps(data, indent=2)}\n"
        f"Resposta completa e detalhada:"
    )

    response = model.generate_content(final_prompt, generation_config=genai.types.GenerationConfig(
        temperature=0.5,
    ))
    return response.text.strip()

# Lógica do chat
if prompt := st.chat_input("Faça sua pergunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Gerando consulta SQL..."):
            try:
                sql_query = get_sql_query_from_llm(prompt)
                sql_query = re.sub(r'^\s*```sql|```\s*$', '', sql_query, flags=re.MULTILINE).strip()
                # st.write(f"SQL gerado: `{sql_query}`")

                with st.spinner("Executando consulta no banco de dados..."):
                    db_result = execute_sql_query(sql_query)
                
                if db_result is None:
                    st.warning("Não foi possível executar a consulta. O fluxo será interrompido.")
                else:
                    with st.spinner("Gerando resposta final..."):
                        final_response = get_final_response_from_llm(prompt, db_result)
                        st.markdown(final_response)
                        st.session_state.messages.append({"role": "assistant", "content": final_response})

            except Exception as e:
                st.error(f"Ocorreu um erro no processo: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Ocorreu um erro: {e}"})