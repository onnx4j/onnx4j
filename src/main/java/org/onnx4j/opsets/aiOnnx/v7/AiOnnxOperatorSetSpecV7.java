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
package org.onnx4j.opsets.aiOnnx.v7;

import java.util.Map;

import org.onnx4j.opsets.Operator;
import org.onnx4j.opsets.aiOnnx.v6.AiOnnxOperatorSetSpecV6;
import org.onnx4j.opsets.aiOnnx.v7.ops.AveragePoolV7;
import org.onnx4j.opsets.aiOnnx.v7.ops.BatchNormalizationV7;
import org.onnx4j.opsets.aiOnnx.v7.ops.DropoutV7;
import org.onnx4j.opsets.aiOnnx.v7.ops.SubV7;

/**
 * Default ONNX Operator Set in version 7
 * 
 * @author HarryLee
 *
 */
public interface AiOnnxOperatorSetSpecV7<T_TENSOR> extends AiOnnxOperatorSetSpecV6<T_TENSOR> {

	// public abstract AcosV7<T_TENSOR> getAcosV7();

	public abstract BatchNormalizationV7<T_TENSOR> getBatchNormalizationV7();

	public abstract DropoutV7<T_TENSOR> getDropoutV7();

	public abstract AveragePoolV7<T_TENSOR> getAveragePoolV7();

	public abstract SubV7<T_TENSOR> getSubV7();

	@Override
	public default Map<String, Operator> initializeOperators() {
		Map<String, Operator> operators = AiOnnxOperatorSetSpecV6.super.initializeOperators();
		// 20191026
		// operators.put(AcosV7.OP_TYPE, this.getAcosV7());
		// 20191029
		operators.put(BatchNormalizationV7.OP_TYPE, this.getBatchNormalizationV7());
		// 20191120
		operators.put(DropoutV7.OP_TYPE, this.getDropoutV7());
		operators.put(AveragePoolV7.OP_TYPE, this.getAveragePoolV7());
		// 20191227
		operators.put(SubV7.OP_TYPE, this.getSubV7());
		return operators;
	}

}