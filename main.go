package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/openai"
	"github.com/firebase/genkit/go/plugins/milvus"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	collectionName := "products"
	idKey := "id"
	vectorKey := "vector"
	vectorDim := int64(1536)
	textKey := "text"
	id1 := int64(1)
	id2 := int64(2)
	id3 := int64(3)
	id4 := int64(4)

	engine, err := milvus.NewEngine(ctx, milvus.WithAddress("127.0.0.1:19530"))
	if err != nil {
		log.Fatalf("failed creating milvus engine: %v", err)
	}
	defer engine.Close(ctx)

	has, err := engine.GetClient().HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		log.Fatalf("failed checking if collection exists: %v", err)
	}

	if !has {
		indexes := []milvusclient.CreateIndexOption{
			milvusclient.NewCreateIndexOption(collectionName, vectorKey,
				index.NewAutoIndex(entity.COSINE)),
		}
		schema := entity.NewSchema().WithDynamicFieldEnabled(true).
			WithField(entity.NewField().
				WithName(idKey).
				WithIsAutoID(false).
				WithDataType(entity.FieldTypeInt64).
				WithIsPrimaryKey(true),
			).
			WithField(entity.NewField().
				WithName(vectorKey).
				WithDataType(entity.FieldTypeFloatVector).
				WithDim(vectorDim),
			).
			WithField(entity.NewField().
				WithName(textKey).
				WithDataType(entity.FieldTypeVarChar).
				WithMaxLength(512),
			)

		if err := engine.GetClient().CreateCollection(ctx,
			milvusclient.NewCreateCollectionOption(collectionName, schema).
				WithIndexOptions(indexes...),
		); err != nil {
			log.Fatalf("failed creating collection: %v", err)
		}
		task, err := engine.GetClient().LoadCollection(ctx,
			milvusclient.NewLoadCollectionOption(collectionName))
		if err != nil {
			log.Fatalf("failed loading collection: %v", err)
		}
		if err := task.Await(ctx); err != nil {
			log.Fatalf("failed waiting for collection load: %v", err)
		}
	}

	_, err = engine.GetClient().Delete(ctx,
		milvusclient.NewDeleteOption(collectionName).WithInt64IDs(idKey, []int64{id1, id2, id3, id4}))
	if err != nil {
		log.Fatalf("failed deleting collection: %v", err)
	}

	m := &milvus.Milvus{Engine: engine}
	o := &openai.OpenAI{}
	g := genkit.Init(ctx, genkit.WithPlugins(m, o))
	e := o.Embedder(g, "text-embedding-3-small")

	colCfg := &milvus.CollectionConfig{
		Name:      collectionName,
		VectorDim: 1536,
		IdKey:     idKey,
		VectorKey: vectorKey,
		TextKey:   textKey,
		ScoreKey:  "score",
		Embedder:  e,
	}

	retOpts := &ai.RetrieverOptions{
		ConfigSchema: core.InferSchemaMap(milvus.RetrieverOptions{}),
		Label:        "",
		Supports: &ai.RetrieverSupports{
			Media: false,
		},
	}

	docStore, retrieval, err := milvus.DefineRetriever(ctx, g, colCfg, retOpts)
	if err != nil {
		log.Fatalf("failed defining retriever: %v", err)
	}

	template := `
		You're a salesman at a phone store. 
		Help the client choose a mobile phone.
		Question: {{question}}
		Context: {{context}}
	`

	prompt := genkit.DefinePrompt(g, "question",
		ai.WithModelName("openai/gpt-4o"),
		ai.WithPrompt(template),
		ai.WithInputType(promptInput{}),
		ai.WithOutputFormat(ai.OutputFormatText),
	)

	flow := genkit.DefineFlow(g, "flow", func(ctx context.Context, query string) (string, error) {
		indexDocs := []*ai.Document{
			ai.DocumentFromText("iPhone 17 $1000", map[string]any{"id": id1}),
			ai.DocumentFromText("Samsung s25 $900", map[string]any{"id": id2}),
			ai.DocumentFromText("Pixel 9 $800", map[string]any{"id": id3}),
			ai.DocumentFromText("Xiaomi 15 $300", map[string]any{"id": id4}),
		}
		if err := milvus.Index(ctx, indexDocs, docStore); err != nil {
			return "", fmt.Errorf("failed indexing documents: %v", err)
		}

		queryDoc := ai.DocumentFromText(query, nil)
		retrieveDocs, err := genkit.Retrieve(ctx, g,
			ai.WithRetriever(retrieval),
			ai.WithDocs(queryDoc),
			ai.WithConfig(&milvus.RetrieverOptions{
				Limit: 2,
			}),
		)
		if err != nil {
			return "", fmt.Errorf("failed retrieving documents: %v", err)
		}

		var sb strings.Builder
		for _, d := range retrieveDocs.Documents {
			sb.WriteString(d.Content[0].Text)
			sb.WriteByte('\n')
		}

		input := &promptInput{
			Question: query,
			Context:  sb.String(),
		}

		promptRes, err := prompt.Execute(ctx, ai.WithInput(input))
		if err != nil {
			return "", fmt.Errorf("failed executing prompt: %v", err)
		}

		return promptRes.Text(), nil
	})

	res, err := flow.Run(ctx, "I want to buy an iphone")
	if err != nil {
		log.Fatalf("failed running flow: %v", err)
	}
	log.Printf("result: %v", res)

	<-ctx.Done()
}

type promptInput struct {
	Question string `json:"question"`
	Context  string `json:"context"`
}
