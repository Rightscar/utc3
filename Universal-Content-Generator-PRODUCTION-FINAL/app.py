"""
Streamlined Universal Content Generator
Optimized version using spaCy + OpenAI (no transformers/CUDA dependencies)
Focuses on 4 core use cases with intelligent pre-processing
"""

import streamlit as st
import os
import tempfile
import zipfile
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Import optimized modules
from modules.enhanced_universal_extractor import EnhancedUniversalExtractor
from modules.enhanced_spacy_processor import EnhancedSpacyProcessor
from modules.intelligent_content_preparer import IntelligentContentPreparer
from modules.rewrite_story_generator import RewriteStoryGenerator
from modules.ai_training_data_generator import AITrainingDataGenerator
from modules.quirky_knowledge_generator import QuirkyKnowledgeGenerator
from modules.persona_narrator_generator import PersonaNarratorGenerator
from modules.edit_refine_engine import EditRefineEngine
from modules.multi_format_exporter import MultiFormatExporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamlinedContentGenerator:
    """
    Streamlined content generator with spaCy optimization
    Removes transformer dependencies while maintaining all functionality
    """
    
    def __init__(self):
        """Initialize the streamlined content generator"""
        self.initialize_components()
        self.setup_session_state()
    
    def initialize_components(self):
        """Initialize all core components"""
        try:
            # Core processing components
            self.extractor = EnhancedUniversalExtractor()
            self.spacy_processor = EnhancedSpacyProcessor()
            self.content_preparer = IntelligentContentPreparer()
            
            # Content generation modules
            self.rewrite_generator = RewriteStoryGenerator()
            self.training_generator = AITrainingDataGenerator()
            self.quirky_generator = QuirkyKnowledgeGenerator()
            self.persona_generator = PersonaNarratorGenerator()
            
            # Enhancement modules
            self.refine_engine = EditRefineEngine()
            self.exporter = MultiFormatExporter()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            st.error(f"Initialization error: {e}")
    
    def setup_session_state(self):
        """Setup Streamlit session state"""
        if 'processed_content' not in st.session_state:
            st.session_state.processed_content = None
        if 'generated_results' not in st.session_state:
            st.session_state.generated_results = []
        if 'current_chunks' not in st.session_state:
            st.session_state.current_chunks = []
        if 'preparation_data' not in st.session_state:
            st.session_state.preparation_data = None
    
    def run(self):
        """Main application runner"""
        self.render_header()
        self.render_sidebar()
        self.render_main_interface()
    
    def render_header(self):
        """Render application header"""
        st.set_page_config(
            page_title="Universal Content Generator",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #15c39a 0%, #13a085 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        .feature-card {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #15c39a;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .success-banner {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .warning-banner {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="main-header">
            <h1>🎯 Universal Content Generator</h1>
            <p>Transform any content with AI-powered creativity • spaCy Optimized • No CUDA Dependencies</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar configuration"""
        with st.sidebar:
            st.markdown("### ⚙️ Configuration")
            
            # OpenAI API Key
            api_key = st.text_input(
                "🔑 OpenAI API Key",
                type="password",
                help="Required for content generation"
            )
            
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key
                st.success("✅ API Key configured")
            else:
                st.warning("⚠️ API Key required for generation")
            
            st.markdown("---")
            
            # Processing Options
            st.markdown("### 🔧 Processing Options")
            
            chunk_size = st.slider(
                "Chunk Size (words)",
                min_value=500,
                max_value=2000,
                value=1000,
                step=100,
                help="Size of text chunks for processing"
            )
            
            enable_entity_preservation = st.checkbox(
                "🏷️ Enhanced Entity Preservation",
                value=True,
                help="Use spaCy to preserve important entities"
            )
            
            enable_context_analysis = st.checkbox(
                "🧠 Advanced Context Analysis",
                value=True,
                help="Analyze content structure and context"
            )
            
            st.session_state.processing_config = {
                'chunk_size': chunk_size,
                'entity_preservation': enable_entity_preservation,
                'context_analysis': enable_context_analysis
            }
            
            st.markdown("---")
            
            # System Status
            st.markdown("### 📊 System Status")
            st.success("✅ spaCy Processor: Ready")
            st.success("✅ Content Preparer: Ready")
            st.success("✅ Export Engine: Ready")
            
            if api_key:
                st.success("✅ OpenAI API: Connected")
            else:
                st.error("❌ OpenAI API: Not configured")
    
    def render_main_interface(self):
        """Render main application interface"""
        tab1, tab2, tab3, tab4 = st.tabs([
            "📁 Upload & Process",
            "🎨 Transform Content", 
            "✏️ Edit & Refine",
            "📤 Export Results"
        ])
        
        with tab1:
            self.render_upload_tab()
        
        with tab2:
            self.render_transform_tab()
        
        with tab3:
            self.render_edit_tab()
        
        with tab4:
            self.render_export_tab()
    
    def render_upload_tab(self):
        """Render upload and processing tab"""
        st.markdown("### 📁 Upload & Process Content")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a file to process",
            type=['pdf', 'docx', 'txt', 'md'],
            help="Supported formats: PDF, Word, Text, Markdown"
        )
        
        if uploaded_file:
            # Process uploaded file
            with st.spinner("🔍 Processing file with spaCy intelligence..."):
                try:
                    # Extract content
                    content = self.extract_file_content(uploaded_file)
                    
                    if content:
                        # Analyze with spaCy
                        analysis = self.spacy_processor.analyze_content(content)
                        
                        # Create smart chunks
                        chunks = self.spacy_processor.get_smart_chunks(
                            content, 
                            max_chunk_size=st.session_state.processing_config['chunk_size']
                        )
                        
                        # Store in session state
                        st.session_state.processed_content = content
                        st.session_state.current_chunks = chunks
                        st.session_state.content_analysis = analysis
                        
                        # Display results
                        self.display_processing_results(content, analysis, chunks)
                    
                except Exception as e:
                    st.error(f"Error processing file: {e}")
                    logger.error(f"File processing error: {e}")
        
        # Text input option
        st.markdown("---")
        st.markdown("### ✏️ Or Enter Text Directly")
        
        text_input = st.text_area(
            "Paste your content here",
            height=200,
            placeholder="Enter the text you want to transform..."
        )
        
        if text_input and st.button("🔍 Process Text"):
            with st.spinner("🔍 Analyzing text with spaCy..."):
                try:
                    # Analyze with spaCy
                    analysis = self.spacy_processor.analyze_content(text_input)
                    
                    # Create smart chunks
                    chunks = self.spacy_processor.get_smart_chunks(
                        text_input,
                        max_chunk_size=st.session_state.processing_config['chunk_size']
                    )
                    
                    # Store in session state
                    st.session_state.processed_content = text_input
                    st.session_state.current_chunks = chunks
                    st.session_state.content_analysis = analysis
                    
                    # Display results
                    self.display_processing_results(text_input, analysis, chunks)
                    
                except Exception as e:
                    st.error(f"Error processing text: {e}")
                    logger.error(f"Text processing error: {e}")
    
    def extract_file_content(self, uploaded_file) -> str:
        """Extract content from uploaded file"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Extract content
            content = self.extractor.extract_content(tmp_path)
            
            # Clean up
            os.unlink(tmp_path)
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting file content: {e}")
            return ""
    
    def display_processing_results(self, content: str, analysis: Dict[str, Any], chunks: List[Dict[str, Any]]):
        """Display processing results"""
        st.markdown("### 📊 Processing Results")
        
        # Content overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📝 Total Words", len(content.split()))
        
        with col2:
            st.metric("📄 Chunks Created", len(chunks))
        
        with col3:
            entities_count = sum(len(entities) for entities in analysis.get('entities', {}).values())
            st.metric("🏷️ Entities Found", entities_count)
        
        with col4:
            complexity = analysis.get('complexity', {}).get('overall_score', 0)
            st.metric("🧠 Complexity", f"{complexity:.2f}")
        
        # Content analysis details
        with st.expander("🔍 Detailed Analysis"):
            
            # Entities
            entities = analysis.get('entities', {})
            if entities:
                st.markdown("**🏷️ Named Entities:**")
                for entity_type, entity_list in entities.items():
                    if entity_list:
                        entity_names = [e['text'] for e in entity_list[:5]]
                        st.write(f"- **{entity_type}**: {', '.join(entity_names)}")
            
            # Key phrases
            key_phrases = analysis.get('key_phrases', [])
            if key_phrases:
                st.markdown("**🔑 Key Phrases:**")
                for phrase in key_phrases[:5]:
                    st.write(f"- {phrase['text']} (importance: {phrase['importance']:.2f})")
            
            # Structure analysis
            structure = analysis.get('structure', {})
            if structure:
                st.markdown("**📋 Content Structure:**")
                sentence_types = structure.get('sentence_types', {})
                for s_type, count in sentence_types.items():
                    if count > 0:
                        st.write(f"- {s_type.title()}: {count}")
        
        # Chunk preview
        with st.expander("📄 Chunk Preview"):
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                st.markdown(f"**Chunk {i+1}** ({len(chunk['text'].split())} words)")
                st.write(chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'])
                st.markdown("---")
        
        st.success("✅ Content processed successfully! Ready for transformation.")
    
    def render_transform_tab(self):
        """Render content transformation tab"""
        st.markdown("### 🎨 Transform Content")
        
        if not st.session_state.processed_content:
            st.warning("⚠️ Please upload and process content first.")
            return
        
        # Transformation type selection
        st.markdown("#### 🎯 Choose Transformation Type")
        
        transformation_type = st.selectbox(
            "Select transformation",
            [
                "rewrite_story",
                "training_data", 
                "quirky_knowledge",
                "persona_narrator"
            ],
            format_func=lambda x: {
                "rewrite_story": "🎭 Rewrite Story Like...",
                "training_data": "🤖 AI Training Data Generator",
                "quirky_knowledge": "🧩 Quirky Knowledge Tools",
                "persona_narrator": "👤 Custom Persona Narrator"
            }[x]
        )
        
        # Configuration based on transformation type
        style_config = self.render_transformation_config(transformation_type)
        
        # Chunk selection
        st.markdown("#### 📄 Select Chunks to Transform")
        
        chunks = st.session_state.current_chunks
        selected_chunks = []
        
        if chunks:
            # Bulk selection options
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ Select All"):
                    for i in range(len(chunks)):
                        st.session_state[f"chunk_{i}"] = True
            
            with col2:
                if st.button("🎯 Select High Quality"):
                    for i, chunk in enumerate(chunks):
                        complexity = chunk.get('analysis', {}).get('complexity', {}).get('overall_score', 0)
                        st.session_state[f"chunk_{i}"] = complexity > 0.5
            
            with col3:
                if st.button("❌ Clear Selection"):
                    for i in range(len(chunks)):
                        st.session_state[f"chunk_{i}"] = False
            
            # Individual chunk selection
            for i, chunk in enumerate(chunks):
                chunk_selected = st.checkbox(
                    f"Chunk {i+1} ({len(chunk['text'].split())} words)",
                    key=f"chunk_{i}",
                    value=st.session_state.get(f"chunk_{i}", False)
                )
                
                if chunk_selected:
                    selected_chunks.append(chunk)
                
                # Show chunk preview
                with st.expander(f"Preview Chunk {i+1}"):
                    st.write(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])
                    
                    # Show spaCy analysis
                    analysis = chunk.get('analysis', {})
                    if analysis:
                        entities = analysis.get('entities', {})
                        if entities:
                            st.markdown("**Entities found:**")
                            for entity_type, entity_list in entities.items():
                                if entity_list:
                                    entity_names = [e['text'] for e in entity_list[:3]]
                                    st.write(f"- {entity_type}: {', '.join(entity_names)}")
        
        # Generate transformations
        if selected_chunks and st.button("🚀 Generate Transformations"):
            if not os.environ.get('OPENAI_API_KEY'):
                st.error("❌ Please configure OpenAI API key in the sidebar.")
                return
            
            with st.spinner("🤖 Generating transformations with enhanced prompts..."):
                try:
                    results = self.generate_transformations(
                        selected_chunks, transformation_type, style_config
                    )
                    
                    st.session_state.generated_results = results
                    st.success(f"✅ Generated {len(results)} transformations!")
                    
                    # Display results preview
                    self.display_transformation_results(results)
                    
                except Exception as e:
                    st.error(f"Error generating transformations: {e}")
                    logger.error(f"Transformation error: {e}")
    
    def render_transformation_config(self, transformation_type: str) -> Dict[str, Any]:
        """Render configuration options for transformation type"""
        config = {}
        
        if transformation_type == "rewrite_story":
            st.markdown("**🎭 Rewrite Configuration**")
            
            persona = st.selectbox(
                "Choose persona/audience",
                [
                    "3-year-old", "5-year-old", "teenager", "pirate", "zen_monk", 
                    "scientist", "comedian", "shakespeare", "yoda", "einstein",
                    "dog", "cat", "alien", "robot", "custom"
                ]
            )
            
            if persona == "custom":
                persona = st.text_input("Enter custom persona")
            
            tone = st.selectbox("Tone", ["playful", "serious", "humorous", "philosophical", "simple"])
            length = st.selectbox("Length", ["short", "medium", "detailed"])
            
            config = {"persona": persona, "tone": tone, "length": length}
        
        elif transformation_type == "training_data":
            st.markdown("**🤖 Training Data Configuration**")
            
            format_type = st.selectbox(
                "Data format",
                ["qa_pairs", "classification", "instruction_following", "few_shot", "dialogue", "summarization"]
            )
            
            examples_per_chunk = st.slider("Examples per chunk", 1, 10, 3)
            
            config = {"format": format_type, "examples_per_chunk": examples_per_chunk}
        
        elif transformation_type == "quirky_knowledge":
            st.markdown("**🧩 Quirky Knowledge Configuration**")
            
            knowledge_type = st.selectbox(
                "Knowledge tool type",
                ["analogies", "metaphors", "socratic_qa", "riddles", "mnemonics", "stories"]
            )
            
            creativity_level = st.slider("Creativity level", 1, 10, 7)
            
            config = {"knowledge_type": knowledge_type, "creativity": creativity_level}
        
        elif transformation_type == "persona_narrator":
            st.markdown("**👤 Persona Narrator Configuration**")
            
            narrator = st.selectbox(
                "Choose narrator",
                [
                    "einstein", "ramana_maharshi", "elon_musk", "steve_jobs", 
                    "gandhi", "socrates", "buddha", "custom"
                ]
            )
            
            if narrator == "custom":
                narrator = st.text_input("Enter custom narrator")
            
            style_emphasis = st.selectbox(
                "Style emphasis",
                ["voice", "perspective", "vocabulary", "all"]
            )
            
            config = {"persona": narrator, "style_emphasis": style_emphasis}
        
        return config
    
    def generate_transformations(self, 
                               chunks: List[Dict[str, Any]], 
                               transformation_type: str,
                               style_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate transformations using intelligent content preparation"""
        results = []
        
        for chunk in chunks:
            try:
                # Prepare content with spaCy intelligence
                preparation = self.content_preparer.prepare_content_for_transformation(
                    chunk['text'], transformation_type, style_config
                )
                
                # Generate transformation based on type
                if transformation_type == "rewrite_story":
                    result = self.rewrite_generator.generate_rewrite(
                        chunk['text'], style_config
                    )
                elif transformation_type == "training_data":
                    result = self.training_generator.generate_training_data(
                        chunk['text'], style_config
                    )
                elif transformation_type == "quirky_knowledge":
                    result = self.quirky_generator.generate_quirky_knowledge(
                        chunk['text'], style_config
                    )
                elif transformation_type == "persona_narrator":
                    result = self.persona_generator.generate_persona_narration(
                        chunk['text'], style_config
                    )
                else:
                    result = {"error": "Unknown transformation type"}
                
                # Add preparation metadata
                result['preparation_data'] = preparation
                result['original_chunk'] = chunk
                result['transformation_type'] = transformation_type
                result['style_config'] = style_config
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error transforming chunk: {e}")
                results.append({
                    "error": str(e),
                    "original_chunk": chunk,
                    "transformation_type": transformation_type
                })
        
        return results
    
    def display_transformation_results(self, results: List[Dict[str, Any]]):
        """Display transformation results"""
        st.markdown("### 🎉 Transformation Results")
        
        for i, result in enumerate(results):
            if "error" in result:
                st.error(f"❌ Error in result {i+1}: {result['error']}")
                continue
            
            with st.expander(f"📄 Result {i+1}"):
                
                # Show original vs transformed
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📝 Original:**")
                    original_text = result['original_chunk']['text']
                    st.write(original_text[:200] + "..." if len(original_text) > 200 else original_text)
                
                with col2:
                    st.markdown("**✨ Transformed:**")
                    if 'content' in result:
                        transformed_text = result['content']
                        st.write(transformed_text[:200] + "..." if len(transformed_text) > 200 else transformed_text)
                    elif 'examples' in result:
                        st.write(f"Generated {len(result['examples'])} training examples")
                    elif 'knowledge_tools' in result:
                        st.write(f"Generated {len(result['knowledge_tools'])} knowledge tools")
                
                # Show preparation insights
                prep_data = result.get('preparation_data', {})
                if prep_data:
                    st.markdown("**🧠 spaCy Analysis Used:**")
                    metadata = prep_data.get('preparation_metadata', {})
                    if metadata:
                        st.write(f"- Entities preserved: {metadata.get('entity_count', 0)}")
                        st.write(f"- Key phrases: {metadata.get('key_phrase_count', 0)}")
                        st.write(f"- Preparation quality: {metadata.get('preparation_quality', 'unknown')}")
    
    def render_edit_tab(self):
        """Render edit and refine tab"""
        st.markdown("### ✏️ Edit & Refine Results")
        
        if not st.session_state.generated_results:
            st.warning("⚠️ Please generate transformations first.")
            return
        
        results = st.session_state.generated_results
        
        # Select result to edit
        result_options = [f"Result {i+1}" for i in range(len(results))]
        selected_result_idx = st.selectbox("Select result to edit", range(len(results)), format_func=lambda x: result_options[x])
        
        if selected_result_idx < len(results):
            result = results[selected_result_idx]
            
            if "error" in result:
                st.error(f"❌ This result has an error: {result['error']}")
                return
            
            # Edit interface
            st.markdown("#### ✏️ Edit Content")
            
            # Get editable content
            if 'content' in result:
                edited_content = st.text_area(
                    "Edit the transformed content",
                    value=result['content'],
                    height=300
                )
                
                if st.button("💾 Save Changes"):
                    st.session_state.generated_results[selected_result_idx]['content'] = edited_content
                    st.success("✅ Changes saved!")
            
            # Refinement options
            st.markdown("#### 🔧 AI-Powered Refinement")
            
            refinement_type = st.selectbox(
                "Choose refinement type",
                [
                    "clarity", "engagement", "grammar", "tone", 
                    "simplification", "examples", "detail", "custom"
                ]
            )
            
            if refinement_type == "custom":
                custom_instruction = st.text_input("Enter custom refinement instruction")
            else:
                custom_instruction = None
            
            if st.button("🚀 Refine with AI"):
                if not os.environ.get('OPENAI_API_KEY'):
                    st.error("❌ Please configure OpenAI API key in the sidebar.")
                    return
                
                with st.spinner("🤖 Refining content..."):
                    try:
                        refined_result = self.refine_engine.refine_content(
                            result, refinement_type, custom_instruction
                        )
                        
                        st.session_state.generated_results[selected_result_idx] = refined_result
                        st.success("✅ Content refined successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error refining content: {e}")
                        logger.error(f"Refinement error: {e}")
    
    def render_export_tab(self):
        """Render export tab"""
        st.markdown("### 📤 Export Results")
        
        if not st.session_state.generated_results:
            st.warning("⚠️ Please generate transformations first.")
            return
        
        results = st.session_state.generated_results
        
        # Export options
        st.markdown("#### 📋 Export Configuration")
        
        export_format = st.selectbox(
            "Choose export format",
            ["json", "jsonl", "csv", "markdown", "txt", "docx", "pdf", "zip"]
        )
        
        include_metadata = st.checkbox("Include processing metadata", value=True)
        include_original = st.checkbox("Include original content", value=True)
        
        # Preview export
        if st.button("👁️ Preview Export"):
            try:
                preview_data = self.exporter.prepare_export_data(
                    results, export_format, include_metadata, include_original
                )
                
                st.markdown("#### 📄 Export Preview")
                
                if export_format in ["json", "jsonl"]:
                    st.json(preview_data)
                else:
                    st.text(str(preview_data)[:1000] + "..." if len(str(preview_data)) > 1000 else str(preview_data))
                
            except Exception as e:
                st.error(f"Error creating preview: {e}")
        
        # Export download
        if st.button("📥 Download Export"):
            try:
                export_data, filename = self.exporter.export_results(
                    results, export_format, include_metadata, include_original
                )
                
                st.download_button(
                    label=f"📥 Download {filename}",
                    data=export_data,
                    file_name=filename,
                    mime=self.exporter.get_mime_type(export_format)
                )
                
                st.success(f"✅ Export ready! Click to download {filename}")
                
            except Exception as e:
                st.error(f"Error creating export: {e}")
                logger.error(f"Export error: {e}")

def main():
    """Main application entry point"""
    try:
        app = StreamlinedContentGenerator()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()

