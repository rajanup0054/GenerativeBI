import streamlit as st
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers



DB_FAISS_PATH = 'vectorstore/db_faiss'

def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=2000,
        temperature=0.5,
        context_length=2000  # Adjust this value as needed
    )
    return llm

tab1, tab2 = st.tabs(["Chat", "Visualize"])

with tab1:


    st.title("Chat with your Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
        data = loader.load()

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

        db = FAISS.from_documents(data, embeddings)
        db.save_local(DB_FAISS_PATH)
        llm = load_llm()
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

        def conversational_chat(query):
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            return result["answer"]

        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Ask me anything about " + uploaded_file.name + " 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey! 👋"]

        response_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = conversational_chat(user_input)

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

with tab2:
    st.title("Visualize Data")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    viz_type = st.sidebar.selectbox("Select Visualization Type", ["Line Chart", "Bar Graph","Heatmap","Scatter Plot","Histogram","Box Plot","Pie Chart"])

    if viz_type == "Line Chart":
        st.subheader("Line Chart")

        x_axis_line = st.selectbox("Select a column for the X-axis", df.columns)
        y_axis_line = st.selectbox("Select a column for the Y-axis", df.columns)

        # Allowing user to filter rows
        rows_line = st.multiselect("Select rows for Line Chart", df.index)

        # Filtering the DataFrame based on user selection
        chart_data_line = df.loc[rows_line, [x_axis_line, y_axis_line]]

        st.write(chart_data_line)

        # Creating a line chart using Altair
        line_chart = alt.Chart(chart_data_line).mark_line().encode(
            x=x_axis_line,
            y=y_axis_line
        )
        st.altair_chart(line_chart, use_container_width=True)

    elif viz_type == "Bar Graph":
        st.subheader("Clustered Bar Chart")

        x_axis_clustered = st.selectbox("Select a column for the X-axis", df.columns)
        y_axis_clustered = st.selectbox("Select a column for the Y-axis", df.columns)

        # Allowing user to filter rows
        rows_bar = st.multiselect("Select rows for Bar Graph", df.index)

        # Filtering the DataFrame based on user selection
        chart_data_bar = df.loc[rows_bar, [x_axis_clustered, y_axis_clustered]]

        st.write(chart_data_bar)

        chart = alt.Chart(chart_data_bar).mark_bar().encode(
            x=x_axis_clustered,
            y=y_axis_clustered
        )
        st.altair_chart(chart, use_container_width=True)

    elif viz_type == "Heatmap":
        st.subheader("Heatmap")

        rows_list = df.index.tolist()
        selected_rows = st.multiselect("Select rows for Heatmap", rows_list)

        columns_list = df.columns.tolist()
        selected_columns = st.multiselect("Select columns for Heatmap", columns_list)

        if selected_rows and selected_columns:
            st.write("Selected Rows:", selected_rows)
            st.write("Selected Columns:", selected_columns)
            heatmap_data = df.loc[selected_rows, selected_columns]

            # Creating a clustered heatmap using Seaborn's clustermap
            sns.set(font_scale=1)  # Adjust font size if needed
            cluster = sns.clustermap(heatmap_data, cmap="rocket", figsize=(10, 8))
            st.pyplot(cluster)
        else:
            st.write("Please select rows and columns for the Heatmap")

    elif viz_type == "Scatter Plot":
        st.subheader("Scatter Plot")

        x_axis = st.selectbox("Select X-axis column", df.columns)
        y_axis = st.selectbox("Select Y-axis column", df.columns)
        size_axis = st.selectbox("Select Size column", df.columns)
        color_axis = st.selectbox("Select Color column", df.columns)

        if x_axis != y_axis and size_axis and color_axis:
            st.write("Selected X-axis:", x_axis)
            st.write("Selected Y-axis:", y_axis)
            st.write("Selected Size column:", size_axis)
            st.write("Selected Color column:", color_axis)

            scatter_chart = alt.Chart(df).mark_circle().encode(
                x=x_axis,
                y=y_axis,
                size=size_axis,
                color=color_axis,
                tooltip=[x_axis, y_axis, size_axis, color_axis]
            ).interactive()

            st.altair_chart(scatter_chart, use_container_width=True)
        else:
            st.write("Please select different columns for X and Y axes, and Size and Color columns")

    elif viz_type == "Histogram":
        st.subheader("Histogram")

        selected_column = st.selectbox("Select a column for Histogram", df.columns)

        if selected_column:
            st.write("Selected Column:", selected_column)

            # Create the histogram based on the selected column
            fig, ax = plt.subplots()
            ax.hist(df[selected_column], bins=20)  # Creating the histogram using plt.hist()

            ax.set_xlabel(selected_column)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Histogram of {selected_column}")

            # Display the plot using Streamlit
            st.pyplot(fig)
        else:
            st.write("Please select a column for the Histogram")
    elif viz_type == "Box Plot":
        st.subheader("Box Plot")

        selected_column = st.selectbox("Select a column for Box Plot", df.columns)

        if selected_column:
            st.write("Selected Column:", selected_column)

            # Check if the selected column contains continuous numeric data
            if pd.api.types.is_numeric_dtype(df[selected_column]):
                # Ask for a column to group by
                group_by_column = st.selectbox("Select a column for grouping", df.columns)

                if group_by_column:
                    st.write("Grouping Column:", group_by_column)

                    # Grouping the data by the group_by_column
                    grouped_data = df.groupby(group_by_column)[selected_column].apply(list).reset_index()

                    # Creating a box plot based on grouped data
                    box_plot = alt.Chart(grouped_data).mark_boxplot().encode(
                        x=group_by_column,
                        y=selected_column
                    ).properties(
                        width=600,
                        height=400
                    )

                    st.altair_chart(box_plot, use_container_width=True)
                else:
                    st.write("Please select a column for grouping")
            else:
                st.write("Selected column is not continuous or numeric")
        else:
            st.write("Please select a column for the Box Plot")

    elif viz_type == "Pie Chart":
        st.subheader("Pie Chart")

        selected_column = st.selectbox("Select a column for Pie Chart", df.columns)

        if selected_column:
            st.write("Selected Column:", selected_column)

            # Count occurrences of each category in the selected column
            pie_data = df[selected_column].value_counts().reset_index()
            pie_data.columns = ['Category', 'Count']

            # Create a pie chart using Altair
            pie_chart = alt.Chart(pie_data).mark_arc().encode(
                color='Category',
                tooltip=['Category', 'Count'],
                theta='Count:Q'
            ).properties(
                width=500,
                height=500
            )

            st.altair_chart(pie_chart, use_container_width=True)
        else:
            st.write("Please select a column for the Pie Chart")