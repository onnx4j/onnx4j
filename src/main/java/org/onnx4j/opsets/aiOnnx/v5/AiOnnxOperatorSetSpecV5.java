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
package org.onnx4j.opsets.aiOnnx.v5;

import java.util.Map;

import org.onnx4j.opsets.Operator;
import org.onnx4j.opsets.aiOnnx.v4.AiOnnxOperatorSetSpecV4;
import org.onnx4j.opsets.aiOnnx.v5.ops.ReshapeV5;

/**
 * Default ONNX Operator Set in version 2
 * 
 * @author HarryLee
 *
 */
public interface AiOnnxOperatorSetSpecV5<T_TENSOR> extends AiOnnxOperatorSetSpecV4<T_TENSOR> {

	public abstract ReshapeV5<T_TENSOR> getReshapeV5();

	//@Override
	public default Map<String, Operator> initializeOperators() {
		Map<String, Operator> operators = AiOnnxOperatorSetSpecV4.super.initializeOperators();
		// 20191120
		operators.put(ReshapeV5.OP_TYPE, this.getReshapeV5());
		return operators;
	}

}