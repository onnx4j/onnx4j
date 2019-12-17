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
package org.onnx4j.opsets.aiOnnx.v4.ops;

import org.onnx4j.opsets.aiOnnx.v1.ops.ConcatV1;
import org.onnx4j.opsets.aiOnnx.v4.AiOnnxOperatorV4;
import org.onnx4j.tensor.DataType;

/**
 * Concat-4
 * 
 * @author HarryLee
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#Concat
 * @version This version of the operator has been available since version 4 of
 *          the default ONNX operator set.
 *
 */
public interface ConcatV4<T_TENSOR> extends ConcatV1<T_TENSOR>, AiOnnxOperatorV4 {

	@Override
	public default DataType[] getTypeConstraints() {
		return DataType.allTypes();
	}

}