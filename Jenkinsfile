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
            // Optional: Add emailext here if SMTP is configured in Jenkins
        }
        failure {
            echo "Pipeline failed for branch ${env.BRANCH_NAME}!"
            // Document requirement: Notify through Email on failure
            emailext (
                subject: "FAILED: Job '${env.JOB_NAME} [Build #${env.BUILD_NUMBER}]'",
                body: "Hi,\n\nThe pipeline has failed on branch ${env.BRANCH_NAME}.\n\nCheck console output at: ${env.BUILD_URL}",
                to: "mabdullah2926@gmail.com" // Aap yahan apni email likh sakte hain
            )
        }
    }
}