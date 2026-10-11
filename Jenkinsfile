pipeline {
    agent any

    stages {
        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Setup Venv & Install Dependencies') {
            steps {
                sh '''
                    python3 -m venv venv
                    ./venv/bin/pip install --upgrade pip
                    ./venv/bin/pip install -r requirements.txt
                '''
            }
        }

        stage('Model Training & MLflow Logging') {
            steps {
                withEnv(["BRANCH_NAME=${env.BRANCH_NAME}"]) {
                    sh './venv/bin/python train.py'
                }
            }
        }
    }

    post {
        success {
            echo "Pipeline for branch ${env.BRANCH_NAME} completed successfully!"
            emailext (
                subject: "SUCCESS: Job '${env.JOB_NAME} [Build #${env.BUILD_NUMBER}]'",
                body: "Hi,\n\nThe pipeline has successfully completed on branch ${env.BRANCH_NAME}.\n\nCheck build details at: ${env.BUILD_URL}",
                to: "mabdullah19027934@gmail.com"
            )
        }
        failure {
            echo "Pipeline failed for branch ${env.BRANCH_NAME}!"
            emailext (
                subject: "FAILED: Job '${env.JOB_NAME} [Build #${env.BUILD_NUMBER}]'",
                body: "Hi,\n\nThe pipeline has failed on branch ${env.BRANCH_NAME}.\n\nCheck console output at: ${env.BUILD_URL}",
                to: "mabdullah19027934@gmail.com"
            )
        }
    }
}