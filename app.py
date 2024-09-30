import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Cargar los datos
data = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Preparar datos para el gráfico de supervivencia por género
gender_survival = data.groupby(['Sex', 'Survived']).size().reset_index(name='Count')
gender_survival['Survived'] = gender_survival['Survived'].replace({0: 'No', 1: 'Sí'})

# Gráfico de barras para la distribución por género y supervivencia
fig_gender_survival = px.bar(
    gender_survival, 
    x='Sex', 
    y='Count', 
    color='Survived', 
    barmode='group', 
    labels={'Sex': 'Género', 'Count': 'Número de pasajeros', 'Survived': 'Sobrevivieron'},
    title='Distribución del género y supervivencia de pasajeros en el Titanic',
    color_discrete_sequence=['#FF4500', '#33FF57']
)

# Preprocesamiento de los datos
data_cleaned = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin']).dropna()

# Codificación de variables categóricas
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

data_cleaned['Sex'] = le_sex.fit_transform(data_cleaned['Sex'])
data_cleaned['Embarked'] = le_embarked.fit_transform(data_cleaned['Embarked'])

# Dividir los datos en entrenamiento y prueba
X = data_cleaned.drop(columns=['Survived'])
y = data_cleaned['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de Regresión Logística
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Calcular la precisión por género
X_test_male = X_test[X_test['Sex'] == 1]
X_test_female = X_test[X_test['Sex'] == 0]

y_test_male = y_test[X_test['Sex'] == 1]
y_test_female = y_test[X_test['Sex'] == 0]

y_pred_male = model.predict(X_test_male)
y_pred_female = model.predict(X_test_female)

accuracy_male = accuracy_score(y_test_male, y_pred_male)
accuracy_female = accuracy_score(y_test_female, y_pred_female)

accuracy_df = pd.DataFrame({
    'Gender': ['Male', 'Female'],
    'Accuracy': [accuracy_male, accuracy_female]
})

# Gráfico de barras para la precisión del modelo por género
fig_accuracy_gender = px.bar(
    accuracy_df, 
    x='Gender', 
    y='Accuracy', 
    title='Precisión del Modelo por Género',
    color_discrete_sequence=['#FF4500'],
    text='Accuracy'  # Mostrar la precisión directamente en las barras
)


# Calcular la media de edad por género 
age_mean_by_gender = data_cleaned.groupby('Sex')['Age'].mean().round(0).reset_index()

# Crear un gráfico de barras para visualizar la media de edad por género
fig_mean_age_by_gender = px.bar(
    age_mean_by_gender, 
    x='Sex',  # Usamos la columna original 'Sex', no la codificada
    y='Age', 
    title='Media de Edad por Género en el Titanic',
    labels={'Age': 'Edad Media', 'Sex': 'Género'},
    text='Age',  # Mostrar la media directamente en la barra
    color_discrete_sequence=['#3498db', '#e74c3c']  # Colores personalizados para cada género
)

# Gráfico para la distribución de edades (histograma)
# Redondear las edades a enteros
data['Age'] = data['Age'].apply(lambda x: int(x) if pd.notnull(x) else x)

fig_age_distribution = px.histogram(
    data, 
    x='Age', 
    nbins=30, 
    title='Distribución de Edades en el Titanic', 
    labels={'Age': 'Edad'}
)

# Inicializar la aplicación Dash
app = dash.Dash(__name__)

# Definir el layout del dashboard con el nuevo gráfico de media de edad
app.layout = html.Div(children=[
    # Título centrado
    html.H1(
        children='Análisis de Sesgos en Datos del Titanic',
        style={
            'textAlign': 'center', 
            'color': '#2c3e50', 
            'backgroundColor': '#ecf0f1', 
            'padding': '10px'
        }
    ),

    # Descripción del dashboard
    html.Div(children='''
        Este dashboard interactivo permite analizar las posibles desigualdades en los datos.
    ''', style={
        'textAlign': 'center', 
        'color': '#7f8c8d', 
        'marginBottom': '20px'
    }),

    # Gráfico de supervivencia por género
    html.Div(
        dcc.Graph(
            id='grafico-supervivencia',
            figure=fig_gender_survival
        ),
        style={
            'backgroundColor': '#ffffff', 
            'padding': '20px', 
            'borderRadius': '10px', 
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 
            'marginBottom': '20px'
        }
    ),

    # Gráfico de precisión por género
    html.Div(
        dcc.Graph(
            id='grafico-precision-genero',
            figure=fig_accuracy_gender
        ),
        style={
            'backgroundColor': '#ffffff', 
            'padding': '20px', 
            'borderRadius': '10px', 
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 
            'marginBottom': '20px',
        }
    ),

    # Gráfico de media de edad por género
    html.Div(
        dcc.Graph(
            id='grafico-media-edad-genero',
            figure=fig_mean_age_by_gender
        ),
        style={
            'backgroundColor': '#ffffff', 
            'padding': '20px', 
            'borderRadius': '10px', 
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 
            'marginBottom': '20px',
        }
    ),

    # Gráfico de distribución de edades
    html.Div(
        dcc.Graph(
            id='grafico-distribucion-edades',
            figure=fig_age_distribution
        ),
        style={
            'backgroundColor': '#ffffff', 
            'padding': '20px', 
            'borderRadius': '10px', 
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)'
        }
    )
], style={
    'backgroundColor': '#ecf0f1', 
    'padding': '50px'
})

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
