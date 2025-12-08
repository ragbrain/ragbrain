"""AWS S3 Vectors provider for serverless vector storage"""

import json
import logging
import time
from typing import List, Dict, Any, Optional, Union
import uuid

import boto3
from botocore.exceptions import ClientError

from ragbrain.providers.base import VectorDBProvider
from ragbrain.config import Settings

logger = logging.getLogger(__name__)


class S3VectorsProvider(VectorDBProvider):
    """AWS S3 Vectors database provider for Lambda-native deployments

    Required IAM permissions:
    - s3vectors:GetVectorBucket
    - s3vectors:GetIndex
    - s3vectors:PutVectors
    - s3vectors:QueryVectors
    - s3vectors:GetVectors (if using metadata filtering or returnMetadata)
    - s3vectors:DeleteVectors
    - s3vectors:ListVectors (for delete_by_metadata operation)
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = boto3.client(
            's3vectors',
            region_name=settings.aws_region
        )
        self.bucket_name = settings.s3vectors_bucket
        self.index_name = settings.s3vectors_index
        self.dimensions = settings.embedding_dimension

        # Validate infrastructure exists (should be pre-provisioned via Terraform/CDK)
        self._validate_infrastructure()

    def _validate_infrastructure(self):
        """Validate that vector bucket and index exist (must be pre-provisioned)"""
        if not self.bucket_name:
            raise ValueError("S3VECTORS_BUCKET environment variable is required")

        try:
            self.client.get_vector_bucket(vectorBucketName=self.bucket_name)
            logger.info(f"S3 Vectors bucket validated: {self.bucket_name}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                raise ValueError(
                    f"S3 Vectors bucket '{self.bucket_name}' not found. "
                    "Please create it using Terraform/CDK before running the application."
                ) from e
            raise

        try:
            self.client.get_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name
            )
            logger.info(f"S3 Vectors index validated: {self.index_name}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                raise ValueError(
                    f"S3 Vectors index '{self.index_name}' not found in bucket '{self.bucket_name}'. "
                    "Please create it using Terraform/CDK before running the application."
                ) from e
            raise

    def _with_retry(self, operation, *args, **kwargs):
        """Execute an S3 Vectors operation with exponential backoff on throttling

        Args:
            operation: Boto3 client method to call
            *args, **kwargs: Arguments to pass to the operation

        Returns:
            Operation result

        Raises:
            ClientError: If the operation fails after retries
        """
        max_retries = 5
        base_delay = 0.5  # Start with 500ms

        for attempt in range(max_retries):
            try:
                return operation(*args, **kwargs)
            except ClientError as e:
                error_code = e.response['Error']['Code']

                # Only retry on throttling errors
                if error_code in ('ThrottlingException', 'TooManyRequestsException'):
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = base_delay * (2 ** attempt)
                        jitter = delay * 0.1  # 10% jitter
                        sleep_time = delay + (jitter * (0.5 - time.time() % 1))

                        logger.warning(
                            f"S3 Vectors throttled, retrying in {sleep_time:.2f}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(sleep_time)
                        continue

                # Re-raise on non-throttling errors or final retry
                raise

    def _validate_metadata(self, text: str, metadata: Dict[str, Any]) -> None:
        """Validate metadata against S3 Vectors limits

        S3 Vectors limitations:
        - Filterable metadata: 2 KB max
        - Total metadata: 40 KB max
        - Metadata keys: 50 max

        Args:
            text: Document text to be stored in metadata
            metadata: User metadata dictionary

        Raises:
            ValueError: If metadata exceeds S3 Vectors limits
        """
        # Count total keys (including 'text' and 'namespace')
        total_keys = len(metadata) + 2
        if total_keys > 50:
            raise ValueError(
                f"Metadata has {total_keys} keys (including 'text' and 'namespace'), "
                f"maximum is 50 per S3 Vectors specification"
            )

        # Build metadata structure to estimate size
        metadata_dict = {
            'text': {'stringValue': text},
            'namespace': {'stringValue': 'default'},  # worst case estimate
            **{
                k: {'stringValue': str(v)} if isinstance(v, str)
                else {'numberValue': v} if isinstance(v, (int, float))
                else {'stringValue': str(v)}
                for k, v in metadata.items()
            }
        }

        # Estimate filterable metadata size (all fields are filterable by default)
        metadata_json = json.dumps(metadata_dict)
        metadata_bytes = len(metadata_json.encode('utf-8'))

        # Warn if approaching 2KB filterable limit
        if metadata_bytes > 2048:  # 2 KB
            logger.warning(
                f"Metadata size is {metadata_bytes} bytes, exceeds S3 Vectors "
                f"filterable metadata limit of 2KB. Request may fail."
            )

        # Check total metadata limit (40 KB)
        if metadata_bytes > 40960:  # 40 KB
            raise ValueError(
                f"Total metadata size is {metadata_bytes} bytes, "
                f"exceeds S3 Vectors limit of 40 KB"
            )

    def insert(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None
    ) -> List[str]:
        """Insert vectors into S3 Vectors"""
        if not ids:
            ids = [str(uuid.uuid4()) for _ in vectors]

        if not metadatas:
            metadatas = [{} for _ in vectors]

        # Build vector records
        vector_records = []
        for id_, vector, text, metadata in zip(ids, vectors, texts, metadatas):
            # Validate metadata against S3 Vectors limits
            self._validate_metadata(text, metadata)

            record = {
                'key': id_,
                'data': {
                    'float32': vector
                },
                'metadata': {
                    'text': {'stringValue': text},
                    'namespace': {'stringValue': namespace or 'default'},
                    **{
                        k: {'stringValue': str(v)} if isinstance(v, str)
                        else {'numberValue': v} if isinstance(v, (int, float))
                        else {'stringValue': str(v)}
                        for k, v in metadata.items()
                    }
                }
            }
            vector_records.append(record)

        # Insert in batches (S3 Vectors supports up to 500 vectors per PutVectors call)
        batch_size = 500
        for i in range(0, len(vector_records), batch_size):
            batch = vector_records[i:i + batch_size]
            self._with_retry(
                self.client.put_vectors,
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                vectors=batch
            )

        logger.info(f"Inserted {len(ids)} vectors into S3 Vectors")
        return ids

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in S3 Vectors

        Supports namespace wildcards:
        - "books/*" matches "books/fiction", "books/nonfiction", etc.
        - "books" matches exactly "books" and sub-namespaces by default
        """
        # S3 Vectors has a hard limit of 100 for topK
        if top_k > 100:
            logger.warning(f"top_k={top_k} exceeds S3 Vectors limit of 100, clamping to 100")
            top_k = 100
        # Build filter expression
        filter_expression = None
        filter_values = {}

        # Handle namespace filtering
        namespace_prefix = None
        exact_namespace = None
        requires_post_filtering = False

        if namespace:
            if namespace.startswith("exact:"):
                # Exact namespace match - use native filter
                exact_namespace = namespace[6:]
                filter_expression = "namespace = :ns"
                filter_values[':ns'] = {'stringValue': exact_namespace}
            elif namespace.endswith("/*"):
                # Wildcard prefix match - requires post-filtering
                # S3 Vectors doesn't support prefix operators, so we need to post-filter
                namespace_prefix = namespace[:-2]
                requires_post_filtering = True
            else:
                # Default behavior: exact match for this namespace
                # This is more efficient than post-filtering for sub-namespaces
                filter_expression = "namespace = :ns"
                filter_values[':ns'] = {'stringValue': namespace}
                exact_namespace = namespace

        # Add custom filters
        if filter:
            conditions = []
            for i, (key, value) in enumerate(filter.items()):
                placeholder = f':v{i}'
                conditions.append(f"{key} = {placeholder}")
                if isinstance(value, str):
                    filter_values[placeholder] = {'stringValue': value}
                elif isinstance(value, (int, float)):
                    filter_values[placeholder] = {'numberValue': value}

            if conditions:
                if filter_expression:
                    filter_expression += " AND " + " AND ".join(conditions)
                else:
                    filter_expression = " AND ".join(conditions)

        # Only increase limit if we need post-filtering for wildcard prefixes
        # Otherwise use exact topK since native filtering is accurate
        search_limit = top_k * 3 if requires_post_filtering else top_k

        # Execute query
        query_params = {
            'vectorBucketName': self.bucket_name,
            'indexName': self.index_name,
            'queryVector': {'float32': query_vector},
            'topK': search_limit
        }

        if filter_expression:
            query_params['filter'] = {
                'expression': filter_expression,
                'expressionAttributeValues': filter_values
            }

        response = self._with_retry(self.client.query_vectors, **query_params)

        results = []
        for match in response.get('vectors', []):
            metadata = match.get('metadata', {})
            ns = self._extract_string_value(metadata.get('namespace', {}))

            # Post-filter for namespace prefix (only for wildcard patterns)
            if requires_post_filtering and namespace_prefix:
                if not (ns == namespace_prefix or ns.startswith(namespace_prefix + "/")):
                    continue

            # Skip document summaries
            doc_type = self._extract_string_value(metadata.get('_type', {}))
            if doc_type == 'document_summary':
                continue

            # Build result
            result = {
                'id': match.get('key'),
                'score': match.get('score', 0.0),
                'content': self._extract_string_value(metadata.get('text', {})),
                'namespace': ns,
                'metadata': {
                    k: self._extract_value(v)
                    for k, v in metadata.items()
                    if k not in ('text', 'namespace', '_type')
                }
            }
            results.append(result)

            if len(results) >= top_k:
                break

        return results

    def _extract_string_value(self, value_obj: Dict) -> str:
        """Extract string value from S3 Vectors metadata format"""
        if isinstance(value_obj, dict):
            return value_obj.get('stringValue', '')
        return str(value_obj) if value_obj else ''

    def _extract_value(self, value_obj: Dict) -> Union[str, float, bool, None]:
        """Extract value from S3 Vectors metadata format

        Args:
            value_obj: S3 Vectors metadata value object

        Returns:
            Extracted primitive value (string, number, boolean, or None)
        """
        if isinstance(value_obj, dict):
            if 'stringValue' in value_obj:
                return value_obj['stringValue']
            if 'numberValue' in value_obj:
                return value_obj['numberValue']
            if 'booleanValue' in value_obj:
                return value_obj['booleanValue']
        return None

    def delete(self, ids: List[str], namespace: Optional[str] = None) -> bool:
        """Delete vectors by IDs"""
        if not ids:
            return True

        try:
            # S3 Vectors supports up to 500 vectors per DeleteVectors call
            batch_size = 500
            for i in range(0, len(ids), batch_size):
                batch = ids[i:i + batch_size]
                self._with_retry(
                    self.client.delete_vectors,
                    vectorBucketName=self.bucket_name,
                    indexName=self.index_name,
                    keys=batch
                )
            logger.info(f"Deleted {len(ids)} vectors from S3 Vectors")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False

    def delete_by_metadata(
        self,
        field: str,
        value: str,
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """Delete vectors by metadata field value"""
        # List vectors matching the criteria
        filter_expression = f"{field} = :val"
        filter_values = {':val': {'stringValue': value}}

        if namespace:
            filter_expression += " AND namespace = :ns"
            filter_values[':ns'] = {'stringValue': namespace}

        # Paginate through all matching vectors
        ids_to_delete = []
        paginator = self.client.get_paginator('list_vectors')

        for page in paginator.paginate(
            vectorBucketName=self.bucket_name,
            indexName=self.index_name,
            filter={
                'expression': filter_expression,
                'expressionAttributeValues': filter_values
            }
        ):
            for vector in page.get('vectors', []):
                ids_to_delete.append(vector.get('key'))

        if not ids_to_delete:
            return {'deleted': 0, 'ids': []}

        # Delete in batches (S3 Vectors supports up to 500 vectors per DeleteVectors call)
        batch_size = 500
        for i in range(0, len(ids_to_delete), batch_size):
            batch = ids_to_delete[i:i + batch_size]
            self._with_retry(
                self.client.delete_vectors,
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                keys=batch
            )

        logger.info(f"Deleted {len(ids_to_delete)} vectors by {field}={value}")
        return {'deleted': len(ids_to_delete), 'ids': ids_to_delete}

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the S3 Vectors index"""
        try:
            index_info = self.client.get_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name
            )
            return {
                'name': self.index_name,
                'bucket': self.bucket_name,
                'dimension': index_info.get('dimension'),
                'distanceMetric': index_info.get('distanceMetric'),
                'status': index_info.get('status')
            }
        except ClientError as e:
            logger.error(f"Failed to get index info: {e}")
            return {
                'name': self.index_name,
                'bucket': self.bucket_name,
                'error': str(e)
            }
