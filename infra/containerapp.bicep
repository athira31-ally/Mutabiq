param location string = resourceGroup().location
param containerAppName string = 'trakheesi-compliance'
param containerImage string
param acrLoginServer string
param pexelsApiKey string = ''
@secure()
param azureVisionKey string = ''
param azureVisionEndpoint string = ''

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: 'trakheesi-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
  }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'trakheesi-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: containerAppName
  location: location
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      secrets: [
        {
          name: 'vision-key'
          value: azureVisionKey
        }
      ]
      ingress: {
        external: true
        targetPort: 8080
      }
    }
    template: {
      containers: [
        {
          name: 'api'
          image: '${acrLoginServer}/${containerImage}'
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            { name: 'APP_ENV', value: 'prod' }
            { name: 'YOLO_ONNX_PATH', value: 'models/yolov8n_watermark.onnx' }
            { name: 'DETECT_CONF_THRESHOLD', value: '0.42' }
            { name: 'PEXELS_API_KEY', value: pexelsApiKey }
            { name: 'AZURE_VISION_ENDPOINT', value: azureVisionEndpoint }
            { name: 'AZURE_VISION_KEY', secretRef: 'vision-key' }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 5
      }
    }
  }
}

output fqdn string = app.properties.configuration.ingress.fqdn
