# 配置管理类
import configparser
import os

class ConfigManager:
    def __init__(self, config_file="config.ini"):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        
    def load_config(self):
        """从配置文件加载配置"""
        if not os.path.exists(self.config_file):
            return None, "配置文件不存在"
            
        try:
            self.config.read(self.config_file)
            return self.config, ""
        except Exception as e:
            return None, str(e)
            
    def save_config(self, config_dict):
        """保存配置到文件"""
        try:
            # 创建新的配置解析器
            self.config = configparser.ConfigParser()
            
            # 将字典转换为ConfigParser格式
            for section, options in config_dict.items():
                self.config[section] = options
                
            # 写入文件
            with open(self.config_file, "w") as configfile:
                self.config.write(configfile)
                
            return True, ""
        except Exception as e:
            return False, str(e)