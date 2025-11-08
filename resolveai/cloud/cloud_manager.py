"""
Cloud Services Manager

Handles integration with various cloud providers for scalable video processing,
storage, and AI model inference.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
import json
import os
from datetime import datetime, timedelta

from ..config.settings import Settings, CloudConfig
from ..security.encryption import EncryptionManager


class CloudProviderError(Exception):
    """Custom exception for cloud provider errors."""
    pass


class CloudProvider(ABC):
    """Abstract base class for cloud providers."""
    
    @abstractmethod
    async def initialize(self, config: CloudConfig) -> None:
        """Initialize the cloud provider."""
        pass
    
    @abstractmethod
    async def upload_file(self, file_path: str, bucket_name: str, object_name: str) -> str:
        """Upload a file to cloud storage."""
        pass
    
    @abstractmethod
    async def download_file(self, bucket_name: str, object_name: str, local_path: str) -> str:
        """Download a file from cloud storage."""
        pass
    
    @abstractmethod
    async def process_video(self, file_path: str, processing_options: Dict[str, Any]) -> Dict[str, Any]:
        """Process a video file using cloud services."""
        pass
    
    @abstractmethod
    async def run_ai_inference(self, model_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run AI model inference."""
        pass
    
    @abstractmethod
    async def cleanup_resources(self) -> None:
        """Clean up cloud resources."""
        pass


class AWSProvider(CloudProvider):
    """Amazon Web Services cloud provider implementation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.s3_client = None
        self.rekognition_client = None
        self.lambda_client = None
        self.ec2_client = None
        self.config = None
    
    async def initialize(self, config: CloudConfig) -> None:
        """Initialize AWS services."""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            self.config = config
            
            # Initialize AWS clients
            session = boto3.Session(
                aws_access_key_id=config.access_key,
                aws_secret_access_key=config.secret_key,
                region_name=config.region
            )
            
            self.s3_client = session.client('s3')
            self.rekognition_client = session.client('rekognition')
            self.lambda_client = session.client('lambda')
            self.ec2_client = session.client('ec2')
            
            # Test connection
            self.s3_client.list_buckets()
            
            self.logger.info("AWS provider initialized successfully")
            
        except ImportError:
            raise CloudProviderError("boto3 is required for AWS provider")
        except Exception as e:
            raise CloudProviderError(f"Failed to initialize AWS: {e}")
    
    async def upload_file(self, file_path: str, bucket_name: str, object_name: str) -> str:
        """Upload file to S3."""
        try:
            self.s3_client.upload_file(file_path, bucket_name, object_name)
            
            # Generate presigned URL
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': object_name},
                ExpiresIn=3600
            )
            
            self.logger.info(f"Uploaded {file_path} to s3://{bucket_name}/{object_name}")
            return url
            
        except Exception as e:
            self.logger.error(f"Failed to upload file: {e}")
            raise CloudProviderError(f"Upload failed: {e}")
    
    async def download_file(self, bucket_name: str, object_name: str, local_path: str) -> str:
        """Download file from S3."""
        try:
            self.s3_client.download_file(bucket_name, object_name, local_path)
            self.logger.info(f"Downloaded s3://{bucket_name}/{object_name} to {local_path}")
            return local_path
            
        except Exception as e:
            self.logger.error(f"Failed to download file: {e}")
            raise CloudProviderError(f"Download failed: {e}")
    
    async def process_video(self, file_path: str, processing_options: Dict[str, Any]) -> Dict[str, Any]:
        """Process video using AWS services."""
        try:
            # Upload to S3
            bucket_name = self.config.bucket_name or "resolveai-processing"
            object_name = f"processing/{os.path.basename(file_path)}"
            
            await self.upload_file(file_path, bucket_name, object_name)
            
            # Start video analysis with Rekognition
            response = self.rekognition_client.start_content_moderation(
                Video={'S3Object': {'Bucket': bucket_name, 'Name': object_name}},
                MinConfidence=processing_options.get('confidence_threshold', 50)
            )
            
            job_id = response['JobId']
            
            # Wait for job completion
            result = await self._wait_for_rekognition_job(job_id)
            
            # Process results
            analysis_result = {
                "job_id": job_id,
                "content_moderation": result,
                "processing_options": processing_options,
                "timestamp": datetime.now().isoformat()
            }
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Video processing failed: {e}")
            raise CloudProviderError(f"Processing failed: {e}")
    
    async def run_ai_inference(self, model_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run AI inference using AWS Lambda or SageMaker."""
        try:
            # For now, use Lambda for simple inference
            payload = {
                "model_name": model_name,
                "input_data": input_data
            }
            
            response = self.lambda_client.invoke(
                FunctionName=f"resolveai-{model_name}",
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            
            result = json.loads(response['Payload'].read())
            
            return {
                "model_name": model_name,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"AI inference failed: {e}")
            raise CloudProviderError(f"Inference failed: {e}")
    
    async def _wait_for_rekognition_job(self, job_id: str, max_wait_time: int = 300) -> Dict[str, Any]:
        """Wait for Rekognition job completion."""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < max_wait_time:
            try:
                response = self.rekognition_client.get_content_moderation(JobId=job_id)
                
                if response['JobStatus'] in ['SUCCEEDED', 'FAILED']:
                    return response
                
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error checking job status: {e}")
                await asyncio.sleep(5)
        
        raise CloudProviderError(f"Job {job_id} timed out")
    
    async def cleanup_resources(self) -> None:
        """Clean up AWS resources."""
        # Implementation depends on specific resources created
        pass


class GCPProvider(CloudProvider):
    """Google Cloud Platform provider implementation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.storage_client = None
        self.video_client = None
        self.config = None
    
    async def initialize(self, config: CloudConfig) -> None:
        """Initialize GCP services."""
        try:
            from google.cloud import storage
            from google.cloud import videointelligence_v1 as videointelligence
            from google.oauth2 import service_account
            
            self.config = config
            
            # Initialize clients
            if config.access_key:
                credentials = service_account.Credentials.from_service_account_file(config.access_key)
                self.storage_client = storage.Client(credentials=credentials, project=config.project_id)
                self.video_client = videointelligence.VideoIntelligenceServiceClient(credentials=credentials)
            else:
                self.storage_client = storage.Client(project=config.project_id)
                self.video_client = videointelligence.VideoIntelligenceServiceClient()
            
            self.logger.info("GCP provider initialized successfully")
            
        except ImportError:
            raise CloudProviderError("google-cloud packages are required for GCP provider")
        except Exception as e:
            raise CloudProviderError(f"Failed to initialize GCP: {e}")
    
    async def upload_file(self, file_path: str, bucket_name: str, object_name: str) -> str:
        """Upload file to Google Cloud Storage."""
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(object_name)
            blob.upload_from_filename(file_path)
            
            # Make it public (or use signed URLs)
            url = blob.public_url
            
            self.logger.info(f"Uploaded {file_path} to gs://{bucket_name}/{object_name}")
            return url
            
        except Exception as e:
            self.logger.error(f"Failed to upload file: {e}")
            raise CloudProviderError(f"Upload failed: {e}")
    
    async def download_file(self, bucket_name: str, object_name: str, local_path: str) -> str:
        """Download file from Google Cloud Storage."""
        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(object_name)
            blob.download_to_filename(local_path)
            
            self.logger.info(f"Downloaded gs://{bucket_name}/{object_name} to {local_path}")
            return local_path
            
        except Exception as e:
            self.logger.error(f"Failed to download file: {e}")
            raise CloudProviderError(f"Download failed: {e}")
    
    async def process_video(self, file_path: str, processing_options: Dict[str, Any]) -> Dict[str, Any]:
        """Process video using Google Cloud Video Intelligence."""
        try:
            # Upload to GCS
            bucket_name = self.config.bucket_name or "resolveai-processing"
            object_name = f"processing/{os.path.basename(file_path)}"
            
            await self.upload_file(file_path, bucket_name, object_name)
            gcs_uri = f"gs://{bucket_name}/{object_name}"
            
            # Start video analysis
            features = ['LABEL_DETECTION', 'SHOT_CHANGE_DETECTION']
            
            if processing_options.get('explicit_content_detection', False):
                features.append('EXPLICIT_CONTENT_DETECTION')
            
            operation = self.video_client.annotate_video(
                input_uri=gcs_uri,
                features=features
            )
            
            # Wait for completion
            result = operation.result()
            
            return {
                "gcs_uri": gcs_uri,
                "analysis_result": result,
                "processing_options": processing_options,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Video processing failed: {e}")
            raise CloudProviderError(f"Processing failed: {e}")
    
    async def run_ai_inference(self, model_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run AI inference using Google Cloud AI Platform."""
        # Implementation would use AI Platform or Vertex AI
        pass
    
    async def cleanup_resources(self) -> None:
        """Clean up GCP resources."""
        pass


class CloudManager:
    """
    Main cloud services manager that coordinates different cloud providers.
    """
    
    def __init__(self, provider: str = "aws", region: str = "us-west-2", 
                 encryption_manager: Optional[EncryptionManager] = None):
        """
        Initialize cloud manager.
        
        Args:
            provider: Cloud provider name (aws, gcp, azure)
            region: Cloud region
            encryption_manager: Optional encryption manager
        """
        self.provider_name = provider.lower()
        self.region = region
        self.encryption_manager = encryption_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize provider
        self.provider = self._create_provider()
        self.settings = Settings()
        self._initialized = False
    
    def _create_provider(self) -> CloudProvider:
        """Create the appropriate cloud provider instance."""
        if self.provider_name == "aws":
            return AWSProvider()
        elif self.provider_name == "gcp":
            return GCPProvider()
        else:
            raise CloudProviderError(f"Unsupported provider: {self.provider_name}")
    
    async def initialize(self) -> None:
        """Initialize the cloud manager."""
        if self._initialized:
            return
        
        try:
            await self.provider.initialize(self.settings.cloud)
            self._initialized = True
            self.logger.info(f"Cloud manager initialized with {self.provider_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cloud manager: {e}")
            raise
    
    async def process_video_file(self, file_path: str, processing_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a video file using cloud services.
        
        Args:
            file_path: Path to the video file
            processing_options: Optional processing configuration
            
        Returns:
            Processing results
        """
        if not self._initialized:
            await self.initialize()
        
        processing_options = processing_options or {}
        
        try:
            # Encrypt file if encryption is enabled
            if self.encryption_manager and self.settings.security.encryption_enabled:
                encrypted_path = self.encryption_manager.encrypt_file(file_path)
                original_path = file_path
                file_path = encrypted_path
                processing_options['encrypted'] = True
                processing_options['original_path'] = original_path
            
            # Process video
            result = await self.provider.process_video(file_path, processing_options)
            
            # Clean up encrypted file if created
            if processing_options.get('encrypted'):
                os.unlink(file_path)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Cloud video processing failed: {e}")
            raise CloudProviderError(f"Processing failed: {e}")
    
    async def upload_to_cloud(self, file_path: str, bucket_name: Optional[str] = None, 
                            object_name: Optional[str] = None) -> str:
        """
        Upload a file to cloud storage.
        
        Args:
            file_path: Path to the file to upload
            bucket_name: Optional bucket name
            object_name: Optional object name
            
        Returns:
            Public URL of the uploaded file
        """
        if not self._initialized:
            await self.initialize()
        
        if not bucket_name:
            bucket_name = self.settings.cloud.bucket_name or "resolveai-storage"
        
        if not object_name:
            object_name = f"uploads/{os.path.basename(file_path)}"
        
        return await self.provider.upload_file(file_path, bucket_name, object_name)
    
    async def download_from_cloud(self, bucket_name: str, object_name: str, 
                                local_path: Optional[str] = None) -> str:
        """
        Download a file from cloud storage.
        
        Args:
            bucket_name: Bucket name
            object_name: Object name
            local_path: Optional local path
            
        Returns:
            Path to the downloaded file
        """
        if not self._initialized:
            await self.initialize()
        
        if not local_path:
            local_path = f"downloads/{os.path.basename(object_name)}"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        return await self.provider.download_file(bucket_name, object_name, local_path)
    
    async def run_cloud_ai_inference(self, model_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run AI model inference in the cloud.
        
        Args:
            model_name: Name of the AI model
            input_data: Input data for the model
            
        Returns:
            Inference results
        """
        if not self._initialized:
            await self.initialize()
        
        return await self.provider.run_ai_inference(model_name, input_data)
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a generic cloud task.
        
        Args:
            task: Task definition
            
        Returns:
            Task results
        """
        task_type = task.get("type")
        
        if task_type == "video_processing":
            return await self.process_video_file(task["file_path"], task.get("options"))
        elif task_type == "upload":
            return await self.upload_to_cloud(task["file_path"], task.get("bucket"), task.get("object"))
        elif task_type == "download":
            return await self.download_from_cloud(task["bucket"], task["object"], task.get("local_path"))
        elif task_type == "ai_inference":
            return await self.run_cloud_ai_inference(task["model"], task["input"])
        else:
            raise CloudProviderError(f"Unknown task type: {task_type}")
    
    async def cleanup(self) -> None:
        """Clean up cloud resources."""
        if self._initialized and self.provider:
            try:
                await self.provider.cleanup_resources()
                self.logger.info("Cloud resources cleaned up")
            except Exception as e:
                self.logger.error(f"Error cleaning up cloud resources: {e}")
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current cloud provider."""
        return {
            "provider": self.provider_name,
            "region": self.region,
            "initialized": self._initialized,
            "encryption_enabled": self.encryption_manager is not None
        }