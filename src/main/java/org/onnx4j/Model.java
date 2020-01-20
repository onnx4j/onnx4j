/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.onnx4j;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

import org.onnx4j.Tensor.Options;
import org.onnx4j.exceptions.ModelException;
import org.onnx4j.exceptions.ModelException.ModelExceptionEnums;
import org.onnx4j.model.Graph;
import org.onnx4j.opsets.OperatorSetId;
import org.onnx4j.prototypes.OnnxProto3;
import org.onnx4j.prototypes.OnnxProto3.ModelProto;
import org.onnx4j.prototypes.OnnxProto3.Version;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Model extends OnnxObject implements AutoCloseable {

	private static Logger logger = LoggerFactory.getLogger(Model.class);

	private long irVersion;
	private long modelVersion;
	private Graph graph;
	private OperatorSetId[] opsetIds;
	private Options tensorOptions;
	private TensorManager<Tensor> tensorManager;

	private static OnnxProto3.ModelProto loadOnnxModel(String onnxModelPath) throws FileNotFoundException, IOException {
		OnnxProto3.ModelProto onnxModel;

		try (InputStream onnxModelInputStream = new FileInputStream(onnxModelPath)) {
			onnxModel = OnnxProto3.ModelProto.parseFrom(onnxModelInputStream);
			assert onnxModel != null;
			logger.info("Model loaded from \"{}\"", onnxModelPath);
			return onnxModel;
		}
	}

	public Model(OnnxProto3.ModelProto onnxModel, Tensor.Options tensorOptions) {
		super(onnxModel.getDocString());

		this.doCheck(onnxModel);
		
		this.tensorManager = new TensorManager<Tensor>() {

			@Override
			protected void dispose(Tensor tensor) {
				tensor.close();
			}
			
		};
		this.tensorOptions = tensorOptions;

		// this.managedTensors = new ManagedTensors();
		this.irVersion = onnxModel.getIrVersion();
		this.modelVersion = onnxModel.getModelVersion();
		this.opsetIds = OperatorSetId.from(onnxModel.getOpsetImportList());
		this.graph = new Graph(this, onnxModel.getGraph());

		super.docString = onnxModel.getDocString();

		if (logger.isDebugEnabled()) {
			String modelInfo = onnxModel.toString();

			if (modelInfo.length() > 3000) {
				modelInfo = String.format("%s ... (omitted %s chars)", modelInfo.substring(0, 3000),
						(modelInfo.length() - 3000));
			}
			logger.debug("{}", modelInfo.replaceAll("[ \n]", ""));
		}
	}

	public Model(String onnxModelPath) throws FileNotFoundException, IOException {
		this(onnxModelPath, Tensor.options());
	}

	public Model(String onnxModelPath, Tensor.Options tensorOptions) throws FileNotFoundException, IOException {
		this(loadOnnxModel(onnxModelPath), tensorOptions);
	}

	public Graph getGraph() {
		return this.graph;
	}

	public long getIrVersion() {
		return this.irVersion;
	}

	public long getModelVersion() {
		return this.modelVersion;
	}

	public OperatorSetId[] getOpsetIds() {
		return this.opsetIds;
	}

	public Options getTensorOptions() {
		return tensorOptions;
	}

	public TensorManager<Tensor> getTensorManager() {
		return tensorManager;
	}
	
	@Override
	public void close() throws Exception {
		this.tensorManager.close();
	}

	/**
	 * 对传入的模型进行必要的合法性检查
	 * 
	 * @param onnxModel
	 */
	private void doCheck(ModelProto onnxModel) {
		//
		// 检查模型的ONNX IR版本是否在支持的范围内
		// 若
		// 模型声明的IR版本号小于ONNX IR定义的最小版本号
		// 或者
		// 模型声明的IR版本号大于于ONNX IR支持的最大版本号
		// 则被认为改模型不被Forwarder支持，抛出Runtime Exception
		//
		long modelIrVersion = onnxModel.getIrVersion();
		if (modelIrVersion > Version.IR_VERSION_VALUE || modelIrVersion < Version._START_VERSION_VALUE)
			throw new ModelException(ModelExceptionEnums.IR_VER_UNSUPPORTED, modelIrVersion, Version.IR_VERSION_VALUE);
	}

}