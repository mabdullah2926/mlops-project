pipeline {
    agent any

    stages {
        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                // WSL/Linux ke liye 'sh' use karein
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Model Training & MLflow Logging') {
            steps {
                // WSL/Linux ke liye 'sh' use karein
                sh 'python3 train.py'
            }
        }
    }

    post {
        failure {
            echo 'Pipeline failed!'
        }
        success {
            echo 'Pipeline completed successfully!'
        }
    }
}