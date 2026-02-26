"""
Derived from dify_dr_workflow
"""
import json
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class FileUtils:
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            Dict[str, Any]: 配置数据
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise

    @staticmethod
    def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
        """保存配置文件
        
        Args:
            config: 配置数据
            config_path: 配置文件路径
        """
        config_path = Path(config_path)
        
        # 确保父目录存在
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
            logger.info(f"Saved configuration to {config_path}")
        
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise

    @staticmethod
    def save_jsonl(data: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
        """保存JSONL格式文件
        
        Args:
            data: 要保存的数据列表
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"Saved {len(data)} items to {output_path}")
        
        except Exception as e:
            logger.error(f"Failed to save JSONL file {output_path}: {e}")
            raise
    
    @staticmethod
    def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """加载JSONL格式文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[Dict[str, Any]]: 数据列表
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {file_path}")
        
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {e}")
            
            logger.info(f"Loaded {len(data)} items from {file_path}")
            return data
        
        except Exception as e:
            logger.error(f"Failed to load JSONL file {file_path}: {e}")
            raise

    @staticmethod
    def save_json(data: Any, output_path: Union[str, Path], indent: int = 2) -> None:
        """保存JSON格式文件
        
        Args:
            data: 要保存的数据
            output_path: 输出文件路径
            indent: 缩进空格数
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
            
            logger.info(f"Saved data to {output_path}")
        
        except Exception as e:
            logger.error(f"Failed to save JSON file {output_path}: {e}")
            raise
    
    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Any:
        """加载JSON格式文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            Any: 数据
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {e}")
            raise