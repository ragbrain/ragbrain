"""Tests for AWS providers (S3 Vectors, DynamoDB, Bedrock)"""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from botocore.exceptions import ClientError

from ragbrain.config import Settings


def make_client_error(code: str, message: str = "Error") -> ClientError:
    """Helper to create ClientError exceptions"""
    return ClientError(
        {"Error": {"Code": code, "Message": message}},
        "TestOperation"
    )


class TestS3VectorsProvider:
    """Tests for S3VectorsProvider"""

    @pytest.fixture
    def mock_settings(self):
        """Create settings for S3 Vectors"""
        return Settings(
            vectordb_provider="s3vectors",
            s3vectors_bucket="test-bucket",
            s3vectors_index="test-index",
            aws_region="us-east-1",
            embedding_dimension=1024
        )

    @pytest.fixture
    def mock_boto_client(self):
        """Create a mock boto3 client"""
        with patch("boto3.client") as mock_client:
            client_instance = MagicMock()
            mock_client.return_value = client_instance
            yield client_instance

    def test_init_validates_infrastructure(self, mock_settings, mock_boto_client):
        """Should validate bucket and index exist on init"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        mock_boto_client.get_vector_bucket.return_value = {}
        mock_boto_client.get_index.return_value = {}

        provider = S3VectorsProvider(mock_settings)

        mock_boto_client.get_vector_bucket.assert_called_once_with(
            vectorBucketName="test-bucket"
        )
        mock_boto_client.get_index.assert_called_once_with(
            vectorBucketName="test-bucket",
            indexName="test-index"
        )

    def test_init_raises_on_missing_bucket(self, mock_settings, mock_boto_client):
        """Should raise ValueError if bucket doesn't exist"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        mock_boto_client.get_vector_bucket.side_effect = make_client_error(
            "ResourceNotFoundException"
        )

        with pytest.raises(ValueError) as exc_info:
            S3VectorsProvider(mock_settings)

        assert "not found" in str(exc_info.value)
        assert "test-bucket" in str(exc_info.value)

    def test_init_raises_on_missing_index(self, mock_settings, mock_boto_client):
        """Should raise ValueError if index doesn't exist"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        mock_boto_client.get_vector_bucket.return_value = {}
        mock_boto_client.get_index.side_effect = make_client_error(
            "ResourceNotFoundException"
        )

        with pytest.raises(ValueError) as exc_info:
            S3VectorsProvider(mock_settings)

        assert "not found" in str(exc_info.value)
        assert "test-index" in str(exc_info.value)

    def test_init_raises_on_missing_bucket_config(self, mock_boto_client):
        """Should raise ValueError if bucket not configured"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        settings = Settings(
            vectordb_provider="s3vectors",
            s3vectors_bucket=None,
            aws_region="us-east-1"
        )

        with pytest.raises(ValueError) as exc_info:
            S3VectorsProvider(settings)

        assert "S3VECTORS_BUCKET" in str(exc_info.value)

    def test_insert_vectors(self, mock_settings, mock_boto_client):
        """Should insert vectors with metadata"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        mock_boto_client.get_vector_bucket.return_value = {}
        mock_boto_client.get_index.return_value = {}

        provider = S3VectorsProvider(mock_settings)

        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        texts = ["text1", "text2"]
        metadatas = [{"key": "value1"}, {"key": "value2"}]

        ids = provider.insert(vectors, texts, metadatas, namespace="test-ns")

        assert len(ids) == 2
        mock_boto_client.put_vectors.assert_called_once()

    def test_search_vectors(self, mock_settings, mock_boto_client):
        """Should search vectors and return results"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        mock_boto_client.get_vector_bucket.return_value = {}
        mock_boto_client.get_index.return_value = {}
        mock_boto_client.query_vectors.return_value = {
            "vectors": [
                {
                    "key": "id1",
                    "score": 0.95,
                    "metadata": {
                        "text": {"stringValue": "test content"},
                        "namespace": {"stringValue": "test-ns"},
                    }
                }
            ]
        }

        provider = S3VectorsProvider(mock_settings)
        results = provider.search([0.1, 0.2, 0.3], top_k=5, namespace="test-ns")

        assert len(results) == 1
        assert results[0]["id"] == "id1"
        assert results[0]["score"] == 0.95
        assert results[0]["content"] == "test content"

    def test_delete_vectors(self, mock_settings, mock_boto_client):
        """Should delete vectors by IDs"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        mock_boto_client.get_vector_bucket.return_value = {}
        mock_boto_client.get_index.return_value = {}

        provider = S3VectorsProvider(mock_settings)
        result = provider.delete(["id1", "id2"])

        assert result is True
        mock_boto_client.delete_vectors.assert_called_once()

    def test_get_collection_info(self, mock_settings, mock_boto_client):
        """Should return index info"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        mock_boto_client.get_vector_bucket.return_value = {}
        mock_boto_client.get_index.return_value = {
            "dimension": 1024,
            "distanceMetric": "cosine",
            "status": "ACTIVE"
        }

        provider = S3VectorsProvider(mock_settings)
        info = provider.get_collection_info()

        assert info["name"] == "test-index"
        assert info["bucket"] == "test-bucket"
        assert info["dimension"] == 1024


class TestDynamoDBNamespaceProvider:
    """Tests for DynamoDBNamespaceProvider"""

    @pytest.fixture
    def mock_settings(self):
        """Create settings for DynamoDB"""
        return Settings(
            namespace_provider="dynamodb",
            dynamodb_namespace_table="test-table",
            aws_region="us-east-1"
        )

    @pytest.fixture
    def mock_dynamodb(self):
        """Create mock DynamoDB resource and client"""
        with patch("boto3.resource") as mock_resource, \
             patch("boto3.client") as mock_client:
            # Mock resource
            resource_instance = MagicMock()
            table_mock = MagicMock()
            resource_instance.Table.return_value = table_mock
            resource_instance.meta.client.meta.region_name = "us-east-1"
            mock_resource.return_value = resource_instance

            # Mock client for describe_table
            client_instance = MagicMock()
            client_instance.describe_table.return_value = {
                "Table": {"TableStatus": "ACTIVE"}
            }
            mock_client.return_value = client_instance

            yield {"resource": resource_instance, "table": table_mock, "client": client_instance}

    def test_init_validates_table(self, mock_settings, mock_dynamodb):
        """Should validate table exists on init"""
        from ragbrain.providers.namespace.dynamodb import DynamoDBNamespaceProvider

        provider = DynamoDBNamespaceProvider(mock_settings)

        mock_dynamodb["client"].describe_table.assert_called_once_with(
            TableName="test-table"
        )

    def test_init_raises_on_missing_table(self, mock_settings, mock_dynamodb):
        """Should raise ValueError if table doesn't exist"""
        from ragbrain.providers.namespace.dynamodb import DynamoDBNamespaceProvider

        mock_dynamodb["client"].describe_table.side_effect = make_client_error(
            "ResourceNotFoundException"
        )

        with pytest.raises(ValueError) as exc_info:
            DynamoDBNamespaceProvider(mock_settings)

        assert "not found" in str(exc_info.value)
        assert "test-table" in str(exc_info.value)

    def test_init_raises_on_inactive_table(self, mock_settings, mock_dynamodb):
        """Should raise ValueError if table is not ACTIVE"""
        from ragbrain.providers.namespace.dynamodb import DynamoDBNamespaceProvider

        mock_dynamodb["client"].describe_table.return_value = {
            "Table": {"TableStatus": "CREATING"}
        }

        with pytest.raises(ValueError) as exc_info:
            DynamoDBNamespaceProvider(mock_settings)

        assert "not ACTIVE" in str(exc_info.value)

    def test_create_namespace(self, mock_settings, mock_dynamodb):
        """Should create a namespace"""
        from ragbrain.providers.namespace.dynamodb import DynamoDBNamespaceProvider

        mock_dynamodb["table"].put_item.return_value = {}

        # First call to exists() check - doesn't exist
        # Second call in create() after put_item - exists
        mock_dynamodb["table"].get_item.side_effect = [
            {},  # exists() check - doesn't exist
            {  # get() after create
                "Item": {
                    "id": "test-ns",
                    "name": "Test Namespace",
                    "description": "A test",
                    "parent_id": "__ROOT__",
                    "metadata": "{}",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z"
                }
            }
        ]

        provider = DynamoDBNamespaceProvider(mock_settings)

        result = provider.create("test-ns", "Test Namespace", "A test")

        assert result["id"] == "test-ns"
        assert result["name"] == "Test Namespace"
        mock_dynamodb["table"].put_item.assert_called_once()

    def test_get_namespace(self, mock_settings, mock_dynamodb):
        """Should get a namespace by ID"""
        from ragbrain.providers.namespace.dynamodb import DynamoDBNamespaceProvider

        mock_dynamodb["table"].get_item.return_value = {
            "Item": {
                "id": "test-ns",
                "name": "Test",
                "description": "",
                "parent_id": "__ROOT__",
                "metadata": "{}",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z"
            }
        }

        provider = DynamoDBNamespaceProvider(mock_settings)
        result = provider.get("test-ns")

        assert result["id"] == "test-ns"
        assert result["parent_id"] is None  # __ROOT__ converted to None

    def test_delete_namespace(self, mock_settings, mock_dynamodb):
        """Should delete a namespace"""
        from ragbrain.providers.namespace.dynamodb import DynamoDBNamespaceProvider

        # exists check
        mock_dynamodb["table"].get_item.return_value = {
            "Item": {"id": "test-ns"}
        }
        # no children
        mock_dynamodb["table"].query.return_value = {"Items": []}

        provider = DynamoDBNamespaceProvider(mock_settings)
        result = provider.delete("test-ns")

        assert result is True
        mock_dynamodb["table"].delete_item.assert_called_once()

    def test_update_namespace_name(self, mock_settings, mock_dynamodb):
        """Should update namespace name with reserved keyword handling"""
        from ragbrain.providers.namespace.dynamodb import DynamoDBNamespaceProvider

        # Mock existing namespace
        existing_item = {
            "id": "test-ns",
            "name": "Old Name",
            "description": "Test",
            "parent_id": "__ROOT__",
            "metadata": "{}",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }

        updated_item = existing_item.copy()
        updated_item["name"] = "New Name"

        mock_dynamodb["table"].get_item.side_effect = [
            {"Item": existing_item},  # First call in update()
            {"Item": updated_item}     # Second call at end of update()
        ]

        provider = DynamoDBNamespaceProvider(mock_settings)
        result = provider.update("test-ns", name="New Name")

        assert result["name"] == "New Name"

        # Verify ExpressionAttributeNames was used for reserved keyword
        call_args = mock_dynamodb["table"].update_item.call_args
        assert call_args.kwargs["ExpressionAttributeNames"]["#n"] == "name"
        assert call_args.kwargs["ExpressionAttributeValues"][":name"] == "New Name"

    def test_update_namespace_description(self, mock_settings, mock_dynamodb):
        """Should update namespace description"""
        from ragbrain.providers.namespace.dynamodb import DynamoDBNamespaceProvider

        existing_item = {
            "id": "test-ns",
            "name": "Test",
            "description": "Old Description",
            "parent_id": "__ROOT__",
            "metadata": "{}",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }

        updated_item = existing_item.copy()
        updated_item["description"] = "New Description"

        mock_dynamodb["table"].get_item.side_effect = [
            {"Item": existing_item},
            {"Item": updated_item}
        ]

        provider = DynamoDBNamespaceProvider(mock_settings)
        result = provider.update("test-ns", description="New Description")

        assert result is not None
        call_args = mock_dynamodb["table"].update_item.call_args
        assert call_args.kwargs["ExpressionAttributeNames"]["#desc"] == "description"

    def test_update_namespace_metadata(self, mock_settings, mock_dynamodb):
        """Should merge metadata when updating"""
        from ragbrain.providers.namespace.dynamodb import DynamoDBNamespaceProvider

        existing_item = {
            "id": "test-ns",
            "name": "Test",
            "description": "",
            "parent_id": "__ROOT__",
            "metadata": '{"key1": "value1"}',
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }

        mock_dynamodb["table"].get_item.side_effect = [
            {"Item": existing_item},
            {"Item": existing_item}
        ]

        provider = DynamoDBNamespaceProvider(mock_settings)
        result = provider.update("test-ns", metadata={"key2": "value2"})

        # Verify metadata was merged
        call_args = mock_dynamodb["table"].update_item.call_args
        import json
        merged_metadata = json.loads(call_args.kwargs["ExpressionAttributeValues"][":meta"])
        assert "key1" in merged_metadata
        assert "key2" in merged_metadata

    def test_update_namespace_parent_validation(self, mock_settings, mock_dynamodb):
        """Should validate parent exists when updating parent_id"""
        from ragbrain.providers.namespace.dynamodb import DynamoDBNamespaceProvider

        existing_item = {
            "id": "test-ns",
            "name": "Test",
            "description": "",
            "parent_id": "__ROOT__",
            "metadata": "{}",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }

        mock_dynamodb["table"].get_item.side_effect = [
            {"Item": existing_item},  # exists check for namespace
            {}                        # parent doesn't exist
        ]

        provider = DynamoDBNamespaceProvider(mock_settings)

        with pytest.raises(ValueError) as exc_info:
            provider.update("test-ns", parent_id="nonexistent-parent")

        assert "Parent namespace not found" in str(exc_info.value)

    def test_update_namespace_self_parent_validation(self, mock_settings, mock_dynamodb):
        """Should prevent namespace from being its own parent"""
        from ragbrain.providers.namespace.dynamodb import DynamoDBNamespaceProvider

        existing_item = {
            "id": "test-ns",
            "name": "Test",
            "description": "",
            "parent_id": "__ROOT__",
            "metadata": "{}",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }

        mock_dynamodb["table"].get_item.return_value = {"Item": existing_item}

        provider = DynamoDBNamespaceProvider(mock_settings)

        with pytest.raises(ValueError) as exc_info:
            provider.update("test-ns", parent_id="test-ns")

        assert "cannot be its own parent" in str(exc_info.value)

    def test_list_namespaces_all(self, mock_settings, mock_dynamodb):
        """Should list all root namespaces when no parent specified"""
        from ragbrain.providers.namespace.dynamodb import DynamoDBNamespaceProvider

        # list() with no args queries for parent_id="__ROOT__", not scan
        mock_dynamodb["table"].query.return_value = {
            "Items": [
                {
                    "id": "ns1",
                    "name": "Namespace 1",
                    "description": "",
                    "parent_id": "__ROOT__",
                    "metadata": "{}",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z"
                },
                {
                    "id": "ns2",
                    "name": "Namespace 2",
                    "description": "",
                    "parent_id": "__ROOT__",
                    "metadata": "{}",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z"
                }
            ]
        }

        provider = DynamoDBNamespaceProvider(mock_settings)
        result = provider.list()

        assert len(result) == 2
        assert result[0]["id"] == "ns1"
        assert result[1]["id"] == "ns2"

    def test_list_namespaces_by_parent(self, mock_settings, mock_dynamodb):
        """Should list namespaces filtered by parent_id"""
        from ragbrain.providers.namespace.dynamodb import DynamoDBNamespaceProvider

        mock_dynamodb["table"].query.return_value = {
            "Items": [
                {
                    "id": "child1",
                    "name": "Child 1",
                    "description": "",
                    "parent_id": "parent-id",
                    "metadata": "{}",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z"
                }
            ]
        }

        provider = DynamoDBNamespaceProvider(mock_settings)
        result = provider.list(parent_id="parent-id")

        assert len(result) == 1
        assert result[0]["id"] == "child1"
        assert result[0]["parent_id"] == "parent-id"

    def test_get_ancestors(self, mock_settings, mock_dynamodb):
        """Should retrieve ancestor chain for a namespace"""
        from ragbrain.providers.namespace.dynamodb import DynamoDBNamespaceProvider

        # Mock hierarchy: root -> parent -> child
        mock_dynamodb["table"].get_item.side_effect = [
            # First call for child
            {
                "Item": {
                    "id": "child",
                    "name": "Child",
                    "parent_id": "parent",
                    "description": "",
                    "metadata": "{}",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z"
                }
            },
            # Second call for parent
            {
                "Item": {
                    "id": "parent",
                    "name": "Parent",
                    "parent_id": "__ROOT__",
                    "description": "",
                    "metadata": "{}",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z"
                }
            }
        ]

        provider = DynamoDBNamespaceProvider(mock_settings)
        ancestors = provider.get_ancestors("child")

        assert len(ancestors) == 1
        assert ancestors[0]["id"] == "parent"

    def test_get_path(self, mock_settings, mock_dynamodb):
        """Should build full path string for namespace"""
        from ragbrain.providers.namespace.dynamodb import DynamoDBNamespaceProvider

        # Mock hierarchy: root -> parent -> child
        mock_dynamodb["table"].get_item.side_effect = [
            # First call in get_ancestors (child)
            {
                "Item": {
                    "id": "child",
                    "name": "Child",
                    "parent_id": "parent",
                    "description": "",
                    "metadata": "{}",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z"
                }
            },
            # Second call in get_ancestors (parent)
            {
                "Item": {
                    "id": "parent",
                    "name": "Parent",
                    "parent_id": "__ROOT__",
                    "description": "",
                    "metadata": "{}",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z"
                }
            },
            # Third call in get_path (child)
            {
                "Item": {
                    "id": "child",
                    "name": "Child",
                    "parent_id": "parent",
                    "description": "",
                    "metadata": "{}",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z"
                }
            }
        ]

        provider = DynamoDBNamespaceProvider(mock_settings)
        path = provider.get_path("child")

        assert path == "Parent > Child"


class TestBedrockLLMProvider:
    """Tests for BedrockLLMProvider"""

    @pytest.fixture
    def mock_settings(self):
        """Create settings for Bedrock LLM"""
        return Settings(
            llm_provider="bedrock",
            bedrock_llm_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            aws_region="us-east-1"
        )

    @pytest.fixture
    def mock_boto_client(self):
        """Create a mock boto3 client"""
        with patch("boto3.client") as mock_client:
            client_instance = MagicMock()
            mock_client.return_value = client_instance
            yield client_instance

    def test_generate_anthropic(self, mock_settings, mock_boto_client):
        """Should generate text using Anthropic format"""
        from ragbrain.providers.llm.bedrock import BedrockLLMProvider

        # Mock response
        response_body = MagicMock()
        response_body.read.return_value = json.dumps({
            "content": [{"text": "Generated response"}]
        })
        mock_boto_client.invoke_model.return_value = {"body": response_body}

        provider = BedrockLLMProvider(mock_settings)
        result = provider.generate("Test prompt")

        assert result == "Generated response"
        mock_boto_client.invoke_model.assert_called_once()

    def test_generate_titan(self, mock_boto_client):
        """Should generate text using Titan format"""
        from ragbrain.providers.llm.bedrock import BedrockLLMProvider

        settings = Settings(
            llm_provider="bedrock",
            bedrock_llm_model="amazon.titan-text-express-v1",
            aws_region="us-east-1"
        )

        response_body = MagicMock()
        response_body.read.return_value = json.dumps({
            "results": [{"outputText": "Titan response"}]
        })
        mock_boto_client.invoke_model.return_value = {"body": response_body}

        provider = BedrockLLMProvider(settings)
        result = provider.generate("Test prompt")

        assert result == "Titan response"

    def test_generate_llama(self, mock_boto_client):
        """Should generate text using Llama format"""
        from ragbrain.providers.llm.bedrock import BedrockLLMProvider

        settings = Settings(
            llm_provider="bedrock",
            bedrock_llm_model="meta.llama3-70b-instruct-v1:0",
            aws_region="us-east-1"
        )

        response_body = MagicMock()
        response_body.read.return_value = json.dumps({
            "generation": "Llama response"
        })
        mock_boto_client.invoke_model.return_value = {"body": response_body}

        provider = BedrockLLMProvider(settings)
        result = provider.generate("Test prompt")

        assert result == "Llama response"

    def test_generate_handles_access_denied(self, mock_settings, mock_boto_client):
        """Should raise RuntimeError on AccessDeniedException"""
        from ragbrain.providers.llm.bedrock import BedrockLLMProvider

        mock_boto_client.invoke_model.side_effect = make_client_error(
            "AccessDeniedException", "Model not enabled"
        )

        provider = BedrockLLMProvider(mock_settings)

        with pytest.raises(RuntimeError) as exc_info:
            provider.generate("Test prompt")

        assert "Access denied" in str(exc_info.value)

    def test_generate_handles_throttling(self, mock_settings, mock_boto_client):
        """Should raise RuntimeError on ThrottlingException"""
        from ragbrain.providers.llm.bedrock import BedrockLLMProvider

        mock_boto_client.invoke_model.side_effect = make_client_error(
            "ThrottlingException"
        )

        provider = BedrockLLMProvider(mock_settings)

        with pytest.raises(RuntimeError) as exc_info:
            provider.generate("Test prompt")

        assert "throttled" in str(exc_info.value)

    def test_generate_handles_validation_error(self, mock_settings, mock_boto_client):
        """Should raise ValueError on ValidationException"""
        from ragbrain.providers.llm.bedrock import BedrockLLMProvider

        mock_boto_client.invoke_model.side_effect = make_client_error(
            "ValidationException", "Invalid input"
        )

        provider = BedrockLLMProvider(mock_settings)

        with pytest.raises(ValueError) as exc_info:
            provider.generate("Test prompt")

        assert "Invalid request" in str(exc_info.value)

    def test_generate_with_context(self, mock_settings, mock_boto_client):
        """Should generate answer with context"""
        from ragbrain.providers.llm.bedrock import BedrockLLMProvider

        response_body = MagicMock()
        response_body.read.return_value = json.dumps({
            "content": [{"text": "Contextual answer"}]
        })
        mock_boto_client.invoke_model.return_value = {"body": response_body}

        provider = BedrockLLMProvider(mock_settings)
        result = provider.generate_with_context(
            "What is X?",
            [{"content": "X is a thing"}]
        )

        assert result == "Contextual answer"


class TestBedrockEmbeddingProvider:
    """Tests for BedrockEmbeddingProvider"""

    @pytest.fixture
    def mock_settings_titan(self):
        """Create settings for Titan embeddings"""
        return Settings(
            embedding_provider="bedrock",
            bedrock_embedding_model="amazon.titan-embed-text-v2:0",
            aws_region="us-east-1"
        )

    @pytest.fixture
    def mock_settings_cohere(self):
        """Create settings for Cohere embeddings on Bedrock"""
        return Settings(
            embedding_provider="bedrock",
            bedrock_embedding_model="cohere.embed-english-v3",
            aws_region="us-east-1"
        )

    @pytest.fixture
    def mock_boto_client(self):
        """Create a mock boto3 client"""
        with patch("boto3.client") as mock_client:
            client_instance = MagicMock()
            mock_client.return_value = client_instance
            yield client_instance

    def test_embed_titan(self, mock_settings_titan, mock_boto_client):
        """Should embed text using Titan"""
        from ragbrain.providers.embeddings.bedrock import BedrockEmbeddingProvider

        response_body = MagicMock()
        response_body.read.return_value = json.dumps({
            "embedding": [0.1, 0.2, 0.3]
        })
        mock_boto_client.invoke_model.return_value = {"body": response_body}

        provider = BedrockEmbeddingProvider(mock_settings_titan)
        result = provider.embed("Test text")

        assert result == [0.1, 0.2, 0.3]

    def test_embed_cohere_document(self, mock_settings_cohere, mock_boto_client):
        """Should embed document text using Cohere with search_document type"""
        from ragbrain.providers.embeddings.bedrock import BedrockEmbeddingProvider

        response_body = MagicMock()
        response_body.read.return_value = json.dumps({
            "embeddings": [[0.1, 0.2, 0.3]]
        })
        mock_boto_client.invoke_model.return_value = {"body": response_body}

        provider = BedrockEmbeddingProvider(mock_settings_cohere)
        result = provider.embed("Test text")

        assert result == [0.1, 0.2, 0.3]

        # Verify input_type was search_document
        call_args = mock_boto_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])
        assert body["input_type"] == "search_document"

    def test_embed_query_cohere(self, mock_settings_cohere, mock_boto_client):
        """Should embed query text using Cohere with search_query type"""
        from ragbrain.providers.embeddings.bedrock import BedrockEmbeddingProvider

        response_body = MagicMock()
        response_body.read.return_value = json.dumps({
            "embeddings": [[0.4, 0.5, 0.6]]
        })
        mock_boto_client.invoke_model.return_value = {"body": response_body}

        provider = BedrockEmbeddingProvider(mock_settings_cohere)
        result = provider.embed_query("Search query")

        assert result == [0.4, 0.5, 0.6]

        # Verify input_type was search_query
        call_args = mock_boto_client.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])
        assert body["input_type"] == "search_query"

    def test_embed_batch_titan(self, mock_settings_titan, mock_boto_client):
        """Should embed batch of texts using Titan (sequential)"""
        from ragbrain.providers.embeddings.bedrock import BedrockEmbeddingProvider

        call_count = [0]

        def mock_invoke(*args, **kwargs):
            call_count[0] += 1
            response_body = MagicMock()
            response_body.read.return_value = json.dumps({
                "embedding": [0.1 * call_count[0]] * 3
            })
            return {"body": response_body}

        mock_boto_client.invoke_model.side_effect = mock_invoke

        provider = BedrockEmbeddingProvider(mock_settings_titan)
        result = provider.embed_batch(["text1", "text2", "text3"])

        assert len(result) == 3
        assert mock_boto_client.invoke_model.call_count == 3

    def test_embed_batch_cohere(self, mock_settings_cohere, mock_boto_client):
        """Should embed batch of texts using Cohere (native batch)"""
        from ragbrain.providers.embeddings.bedrock import BedrockEmbeddingProvider

        response_body = MagicMock()
        response_body.read.return_value = json.dumps({
            "embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        })
        mock_boto_client.invoke_model.return_value = {"body": response_body}

        provider = BedrockEmbeddingProvider(mock_settings_cohere)
        result = provider.embed_batch(["text1", "text2", "text3"])

        assert len(result) == 3
        # Cohere uses native batch, so only one call
        assert mock_boto_client.invoke_model.call_count == 1

    def test_get_dimensions_titan_v2(self, mock_settings_titan, mock_boto_client):
        """Should return correct dimensions for Titan v2"""
        from ragbrain.providers.embeddings.bedrock import BedrockEmbeddingProvider

        provider = BedrockEmbeddingProvider(mock_settings_titan)
        assert provider.get_dimensions() == 1024

    def test_get_dimensions_cohere(self, mock_settings_cohere, mock_boto_client):
        """Should return correct dimensions for Cohere"""
        from ragbrain.providers.embeddings.bedrock import BedrockEmbeddingProvider

        provider = BedrockEmbeddingProvider(mock_settings_cohere)
        assert provider.get_dimensions() == 1024


class TestEmbedQueryBaseClass:
    """Tests for embed_query in base EmbeddingProvider"""

    def test_embed_query_defaults_to_embed(self):
        """Base class embed_query should call embed by default"""
        from ragbrain.providers.base import EmbeddingProvider

        class TestProvider(EmbeddingProvider):
            def __init__(self):
                self.embed_called_with = None

            def embed(self, text):
                self.embed_called_with = text
                return [0.1, 0.2, 0.3]

            def embed_batch(self, texts):
                return [[0.1, 0.2, 0.3] for _ in texts]

            def get_dimensions(self):
                return 3

        provider = TestProvider()
        result = provider.embed_query("test query")

        assert result == [0.1, 0.2, 0.3]
        assert provider.embed_called_with == "test query"


class TestCohereProviderEmbedQuery:
    """Tests for embed_query in Cohere provider"""

    @pytest.fixture
    def mock_cohere_client(self):
        """Create a mock Cohere client"""
        with patch("cohere.Client") as mock_client:
            client_instance = MagicMock()
            mock_client.return_value = client_instance
            yield client_instance

    def test_embed_uses_search_document(self, mock_cohere_client):
        """embed() should use input_type=search_document"""
        from ragbrain.providers.embeddings.cohere import CohereEmbeddingProvider

        mock_cohere_client.embed.return_value = MagicMock(
            embeddings=[[0.1, 0.2, 0.3]]
        )

        settings = Settings(
            embedding_provider="cohere",
            cohere_api_key="test-key"
        )
        provider = CohereEmbeddingProvider(settings)
        provider.embed("test document")

        mock_cohere_client.embed.assert_called_once()
        call_kwargs = mock_cohere_client.embed.call_args.kwargs
        assert call_kwargs["input_type"] == "search_document"

    def test_embed_query_uses_search_query(self, mock_cohere_client):
        """embed_query() should use input_type=search_query"""
        from ragbrain.providers.embeddings.cohere import CohereEmbeddingProvider

        mock_cohere_client.embed.return_value = MagicMock(
            embeddings=[[0.1, 0.2, 0.3]]
        )

        settings = Settings(
            embedding_provider="cohere",
            cohere_api_key="test-key"
        )
        provider = CohereEmbeddingProvider(settings)
        provider.embed_query("search query")

        mock_cohere_client.embed.assert_called_once()
        call_kwargs = mock_cohere_client.embed.call_args.kwargs
        assert call_kwargs["input_type"] == "search_query"


class TestS3VectorsMetadataValidation:
    """Tests for S3 Vectors metadata validation"""

    @pytest.fixture
    def mock_settings(self):
        return Settings(
            vectordb_provider="s3vectors",
            s3vectors_bucket="test-bucket",
            s3vectors_index="test-index",
            aws_region="us-east-1"
        )

    @pytest.fixture
    def mock_boto_client(self):
        with patch("boto3.client") as mock_client:
            client_instance = MagicMock()
            client_instance.get_vector_bucket.return_value = {}
            client_instance.get_index.return_value = {}
            mock_client.return_value = client_instance
            yield client_instance

    def test_metadata_key_limit_exceeded(self, mock_settings, mock_boto_client):
        """Should raise ValueError when metadata exceeds 50 key limit"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        provider = S3VectorsProvider(mock_settings)

        # Create metadata with 49 keys (+ text + namespace = 51 total)
        large_metadata = {f"key{i}": f"value{i}" for i in range(49)}

        with pytest.raises(ValueError) as exc_info:
            provider.insert(
                vectors=[[0.1, 0.2]],
                texts=["test"],
                metadatas=[large_metadata]
            )

        assert "51 keys" in str(exc_info.value)
        assert "maximum is 50" in str(exc_info.value)

    def test_metadata_40kb_limit_exceeded(self, mock_settings, mock_boto_client):
        """Should raise ValueError when metadata exceeds 40KB total limit"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        provider = S3VectorsProvider(mock_settings)

        # Create very large text that exceeds 40KB
        large_text = "x" * 50000  # 50KB text

        with pytest.raises(ValueError) as exc_info:
            provider.insert(
                vectors=[[0.1, 0.2]],
                texts=[large_text],
                metadatas=[{}]
            )

        assert "exceeds S3 Vectors limit of 40 KB" in str(exc_info.value)

    def test_metadata_2kb_warning(self, mock_settings, mock_boto_client, caplog):
        """Should log warning when metadata exceeds 2KB filterable limit"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider
        import logging

        provider = S3VectorsProvider(mock_settings)

        # Create metadata that's >2KB but <40KB
        medium_text = "x" * 3000  # 3KB text

        with caplog.at_level(logging.WARNING):
            provider.insert(
                vectors=[[0.1, 0.2]],
                texts=[medium_text],
                metadatas=[{}]
            )

        assert "exceeds S3 Vectors" in caplog.text
        assert "2KB" in caplog.text

    def test_metadata_within_limits(self, mock_settings, mock_boto_client):
        """Should succeed when metadata is within all limits"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        provider = S3VectorsProvider(mock_settings)

        # Small metadata - should pass
        metadata = {"key1": "value1", "key2": 123}
        ids = provider.insert(
            vectors=[[0.1, 0.2]],
            texts=["small text"],
            metadatas=[metadata]
        )

        assert len(ids) == 1
        mock_boto_client.put_vectors.assert_called_once()


class TestS3VectorsRetryLogic:
    """Tests for S3 Vectors exponential backoff retry logic"""

    @pytest.fixture
    def mock_settings(self):
        return Settings(
            vectordb_provider="s3vectors",
            s3vectors_bucket="test-bucket",
            s3vectors_index="test-index",
            aws_region="us-east-1"
        )

    @pytest.fixture
    def mock_boto_client(self):
        with patch("boto3.client") as mock_client:
            client_instance = MagicMock()
            client_instance.get_vector_bucket.return_value = {}
            client_instance.get_index.return_value = {}
            mock_client.return_value = client_instance
            yield client_instance

    def test_retry_on_throttling_exception(self, mock_settings, mock_boto_client):
        """Should retry on ThrottlingException and eventually succeed"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        provider = S3VectorsProvider(mock_settings)

        # Fail twice, then succeed
        call_count = [0]

        def mock_put_vectors(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise make_client_error("ThrottlingException")
            return {}

        mock_boto_client.put_vectors.side_effect = mock_put_vectors

        with patch("time.sleep"):  # Mock sleep to speed up test
            ids = provider.insert(
                vectors=[[0.1, 0.2]],
                texts=["test"],
                metadatas=[{}]
            )

        assert len(ids) == 1
        assert mock_boto_client.put_vectors.call_count == 3

    def test_retry_on_too_many_requests(self, mock_settings, mock_boto_client):
        """Should retry on TooManyRequestsException"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        provider = S3VectorsProvider(mock_settings)

        call_count = [0]

        def mock_put_vectors(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise make_client_error("TooManyRequestsException")
            return {}

        mock_boto_client.put_vectors.side_effect = mock_put_vectors

        with patch("time.sleep"):
            ids = provider.insert(
                vectors=[[0.1, 0.2]],
                texts=["test"],
                metadatas=[{}]
            )

        assert len(ids) == 1
        assert mock_boto_client.put_vectors.call_count == 2

    def test_no_retry_on_other_errors(self, mock_settings, mock_boto_client):
        """Should not retry on non-throttling errors"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        provider = S3VectorsProvider(mock_settings)

        mock_boto_client.put_vectors.side_effect = make_client_error(
            "ValidationException", "Invalid input"
        )

        with pytest.raises(ClientError):
            provider.insert(
                vectors=[[0.1, 0.2]],
                texts=["test"],
                metadatas=[{}]
            )

        # Should only be called once (no retry)
        assert mock_boto_client.put_vectors.call_count == 1

    def test_max_retries_exceeded(self, mock_settings, mock_boto_client):
        """Should raise error after max retries exceeded"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        provider = S3VectorsProvider(mock_settings)

        # Always fail with throttling
        mock_boto_client.put_vectors.side_effect = make_client_error("ThrottlingException")

        with patch("time.sleep"):
            with pytest.raises(ClientError) as exc_info:
                provider.insert(
                    vectors=[[0.1, 0.2]],
                    texts=["test"],
                    metadatas=[{}]
                )

        assert exc_info.value.response['Error']['Code'] == "ThrottlingException"
        # Should try 5 times (initial + 4 retries)
        assert mock_boto_client.put_vectors.call_count == 5


class TestS3VectorsTopKValidation:
    """Tests for top_k validation and clamping"""

    @pytest.fixture
    def mock_settings(self):
        return Settings(
            vectordb_provider="s3vectors",
            s3vectors_bucket="test-bucket",
            s3vectors_index="test-index",
            aws_region="us-east-1"
        )

    @pytest.fixture
    def mock_boto_client(self):
        with patch("boto3.client") as mock_client:
            client_instance = MagicMock()
            client_instance.get_vector_bucket.return_value = {}
            client_instance.get_index.return_value = {}
            client_instance.query_vectors.return_value = {"vectors": []}
            mock_client.return_value = client_instance
            yield client_instance

    def test_top_k_clamped_to_100(self, mock_settings, mock_boto_client, caplog):
        """Should clamp top_k to 100 and log warning"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider
        import logging

        provider = S3VectorsProvider(mock_settings)

        with caplog.at_level(logging.WARNING):
            provider.search([0.1, 0.2], top_k=150)

        assert "exceeds S3 Vectors limit of 100" in caplog.text
        assert "clamping to 100" in caplog.text

        # Verify the actual query used top_k=100
        call_args = mock_boto_client.query_vectors.call_args
        assert call_args.kwargs['topK'] == 100

    def test_top_k_under_limit(self, mock_settings, mock_boto_client):
        """Should not clamp when top_k is under 100"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        provider = S3VectorsProvider(mock_settings)
        provider.search([0.1, 0.2], top_k=50)

        call_args = mock_boto_client.query_vectors.call_args
        assert call_args.kwargs['topK'] == 50

    def test_top_k_exactly_100(self, mock_settings, mock_boto_client, caplog):
        """Should accept top_k=100 without warning"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider
        import logging

        provider = S3VectorsProvider(mock_settings)

        with caplog.at_level(logging.WARNING):
            provider.search([0.1, 0.2], top_k=100)

        # No warning should be logged
        assert "clamping" not in caplog.text

        call_args = mock_boto_client.query_vectors.call_args
        assert call_args.kwargs['topK'] == 100


class TestS3VectorsNamespaceFiltering:
    """Tests for namespace filtering optimizations"""

    @pytest.fixture
    def mock_settings(self):
        return Settings(
            vectordb_provider="s3vectors",
            s3vectors_bucket="test-bucket",
            s3vectors_index="test-index",
            aws_region="us-east-1"
        )

    @pytest.fixture
    def mock_boto_client(self):
        with patch("boto3.client") as mock_client:
            client_instance = MagicMock()
            client_instance.get_vector_bucket.return_value = {}
            client_instance.get_index.return_value = {}
            client_instance.query_vectors.return_value = {"vectors": []}
            mock_client.return_value = client_instance
            yield client_instance

    def test_exact_namespace_uses_native_filter(self, mock_settings, mock_boto_client):
        """Should use native filter for exact: prefix"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        provider = S3VectorsProvider(mock_settings)
        provider.search([0.1, 0.2], top_k=5, namespace="exact:books")

        call_args = mock_boto_client.query_vectors.call_args
        assert 'filter' in call_args.kwargs
        assert call_args.kwargs['filter']['expression'] == "namespace = :ns"
        assert call_args.kwargs['filter']['expressionAttributeValues'][':ns'] == {'stringValue': 'books'}
        # Should not overfetch for exact match
        assert call_args.kwargs['topK'] == 5

    def test_wildcard_namespace_requires_post_filtering(self, mock_settings, mock_boto_client):
        """Should overfetch and post-filter for wildcard patterns"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        provider = S3VectorsProvider(mock_settings)
        provider.search([0.1, 0.2], top_k=5, namespace="books/*")

        call_args = mock_boto_client.query_vectors.call_args
        # Should overfetch for post-filtering
        assert call_args.kwargs['topK'] == 15  # 5 * 3
        # No filter expression for wildcard (post-filter instead)
        assert 'filter' not in call_args.kwargs

    def test_default_namespace_uses_exact_match(self, mock_settings, mock_boto_client):
        """Should use exact match filter for default namespace behavior"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        provider = S3VectorsProvider(mock_settings)
        provider.search([0.1, 0.2], top_k=5, namespace="books")

        call_args = mock_boto_client.query_vectors.call_args
        assert 'filter' in call_args.kwargs
        assert call_args.kwargs['filter']['expression'] == "namespace = :ns"
        assert call_args.kwargs['topK'] == 5  # No overfetch for exact match

    def test_no_namespace_no_filter(self, mock_settings, mock_boto_client):
        """Should not add filter when no namespace specified"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        provider = S3VectorsProvider(mock_settings)
        provider.search([0.1, 0.2], top_k=5)

        call_args = mock_boto_client.query_vectors.call_args
        assert 'filter' not in call_args.kwargs
        assert call_args.kwargs['topK'] == 5


class TestS3VectorsBatchOperations:
    """Tests for batch delete operations"""

    @pytest.fixture
    def mock_settings(self):
        return Settings(
            vectordb_provider="s3vectors",
            s3vectors_bucket="test-bucket",
            s3vectors_index="test-index",
            aws_region="us-east-1"
        )

    @pytest.fixture
    def mock_boto_client(self):
        with patch("boto3.client") as mock_client:
            client_instance = MagicMock()
            client_instance.get_vector_bucket.return_value = {}
            client_instance.get_index.return_value = {}
            mock_client.return_value = client_instance
            yield client_instance

    def test_delete_small_batch(self, mock_settings, mock_boto_client):
        """Should delete small batch in single call"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        provider = S3VectorsProvider(mock_settings)
        result = provider.delete(["id1", "id2", "id3"])

        assert result is True
        mock_boto_client.delete_vectors.assert_called_once()

    def test_delete_large_batch(self, mock_settings, mock_boto_client):
        """Should split large batch into multiple calls"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        provider = S3VectorsProvider(mock_settings)

        # Create 1000 IDs (should result in 2 batches of 500)
        ids = [f"id{i}" for i in range(1000)]
        result = provider.delete(ids)

        assert result is True
        assert mock_boto_client.delete_vectors.call_count == 2

    def test_delete_empty_list(self, mock_settings, mock_boto_client):
        """Should handle empty list gracefully"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        provider = S3VectorsProvider(mock_settings)
        result = provider.delete([])

        assert result is True
        mock_boto_client.delete_vectors.assert_not_called()


class TestS3VectorsDeleteByMetadata:
    """Tests for S3 Vectors delete_by_metadata operation"""

    @pytest.fixture
    def mock_settings(self):
        return Settings(
            vectordb_provider="s3vectors",
            s3vectors_bucket="test-bucket",
            s3vectors_index="test-index",
            aws_region="us-east-1"
        )

    @pytest.fixture
    def mock_boto_client(self):
        with patch("boto3.client") as mock_client:
            client_instance = MagicMock()
            client_instance.get_vector_bucket.return_value = {}
            client_instance.get_index.return_value = {}
            mock_client.return_value = client_instance
            yield client_instance

    def test_delete_by_metadata_with_results(self, mock_settings, mock_boto_client):
        """Should delete vectors matching metadata criteria"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        provider = S3VectorsProvider(mock_settings)

        # Mock paginator
        paginator = MagicMock()
        mock_boto_client.get_paginator.return_value = paginator

        # Mock pagination results
        paginator.paginate.return_value = [
            {
                "vectors": [
                    {"key": "id1"},
                    {"key": "id2"},
                    {"key": "id3"}
                ]
            }
        ]

        result = provider.delete_by_metadata("doc_type", "article")

        assert result["deleted"] == 3
        assert len(result["ids"]) == 3
        assert "id1" in result["ids"]

        # Should batch delete with 500 limit
        mock_boto_client.delete_vectors.assert_called_once()

    def test_delete_by_metadata_large_batch(self, mock_settings, mock_boto_client):
        """Should handle large batches with multiple delete calls"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        provider = S3VectorsProvider(mock_settings)

        # Mock paginator
        paginator = MagicMock()
        mock_boto_client.get_paginator.return_value = paginator

        # Mock 1000 vectors to delete (should result in 2 batch calls)
        vectors = [{"key": f"id{i}"} for i in range(1000)]
        paginator.paginate.return_value = [{"vectors": vectors}]

        result = provider.delete_by_metadata("status", "archived")

        assert result["deleted"] == 1000
        # Should split into 2 batches of 500
        assert mock_boto_client.delete_vectors.call_count == 2

    def test_delete_by_metadata_no_results(self, mock_settings, mock_boto_client):
        """Should handle case when no vectors match criteria"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        provider = S3VectorsProvider(mock_settings)

        # Mock paginator with no results
        paginator = MagicMock()
        mock_boto_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{"vectors": []}]

        result = provider.delete_by_metadata("status", "nonexistent")

        assert result["deleted"] == 0
        assert result["ids"] == []
        mock_boto_client.delete_vectors.assert_not_called()

    def test_delete_by_metadata_with_namespace(self, mock_settings, mock_boto_client):
        """Should filter by namespace when provided"""
        from ragbrain.providers.vectordb.s3vectors import S3VectorsProvider

        provider = S3VectorsProvider(mock_settings)

        # Mock paginator
        paginator = MagicMock()
        mock_boto_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{"vectors": [{"key": "id1"}]}]

        result = provider.delete_by_metadata("type", "temp", namespace="test-ns")

        assert result["deleted"] == 1

        # Verify filter expression included namespace
        call_args = paginator.paginate.call_args
        filter_expr = call_args.kwargs["filter"]
        assert "namespace = :ns" in filter_expr["expression"]
        assert filter_expr["expressionAttributeValues"][":ns"] == {"stringValue": "test-ns"}
